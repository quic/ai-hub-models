# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime
import pkg_resources
import torch

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.onnx_helpers import (
    extract_io_types_from_onnx_model,
    kwargs_to_dict,
)

ONNXRUNTIME_ENV_CHECKED: bool = False
ONNXRUNTIME_QNN_ERROR: ValueError | None = None


def _hash_dataclass(
    cls: object, ignore_fields: list[str] = [], hash: hashlib._Hash | None = None
) -> hashlib._Hash:
    hash = hash or hashlib.md5()
    for field in dataclasses.fields(cls):  # type: ignore[arg-type]
        if field.name in ignore_fields:
            continue
        hash.update(
            bytes(f"{field.name}: {str(getattr(cls, field.name))}", encoding="utf-8")
        )
    return hash


def _hash_file(
    path: Path | os.PathLike | str, hash: hashlib._Hash | None = None
) -> hashlib._Hash:
    hash = hash or hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hash.update(chunk)
    return hash


def _verify_onnxruntime_qnn_installed() -> None:
    """
    Throws an exception if onnxruntime-qnn:
        * is not installed
        * has conflicts with other installed packages

    Runs only once then caches the result for this python session.
    """
    global ONNXRUNTIME_ENV_CHECKED
    global ONNXRUNTIME_QNN_ERROR
    if ONNXRUNTIME_ENV_CHECKED:
        if ONNXRUNTIME_QNN_ERROR:
            raise ONNXRUNTIME_QNN_ERROR
        return

    pkgs = cast(pkg_resources.WorkingSet, pkg_resources.working_set)
    pkg_names = {cast(str, p.key) for p in pkgs}

    ORT_QNN_PACKAGE_NAME = "onnxruntime-qnn"
    ORT_PACKAGE_NAME = "onnxruntime"
    ORT_GPU_PACKAGE_NAME = "onnxruntime-gpu"
    ORT_DML_PACKAGE_NAME = "onnxruntime-directml"

    ALL_ADDITIONAL_RUNTIMES = [
        ORT_PACKAGE_NAME,
        ORT_GPU_PACKAGE_NAME,
        ORT_DML_PACKAGE_NAME,
    ]
    additional_runtimes_installed = [
        x for x in ALL_ADDITIONAL_RUNTIMES if x in pkg_names
    ]
    qnn_runtime_installed = ORT_QNN_PACKAGE_NAME in pkg_names

    install_instructions: str | None = None
    if os.name != "nt" or "Qualcomm" not in platform.processor():
        install_instructions = (
            "NPU execution is supported only on Windows on Snapdragon devices."
        )
    elif additional_runtimes_installed:
        install_instructions = "\n".join(
            [
                "You're targeting QNN, but have additional onnxruntime packages installed. Only 1 onnxruntime package can be installed at once.",
                "Run the following commands EXACTLY (just copy paste; DO NOT EDIT):",
                f'    pip uninstall -y {" ".join(additional_runtimes_installed)}{f" {ORT_QNN_PACKAGE_NAME}" if qnn_runtime_installed else ""}',
                f"    pip install {ORT_QNN_PACKAGE_NAME}",
            ]
        )
    elif not qnn_runtime_installed:
        install_instructions = "\n".join(
            [
                "You must have onnxruntime-qnn installed to run on NPU:",
                f"    pip install {ORT_QNN_PACKAGE_NAME}",
            ]
        )

    ONNXRUNTIME_ENV_CHECKED = True
    if install_instructions:
        ONNXRUNTIME_QNN_ERROR = ValueError(install_instructions)
        raise ONNXRUNTIME_QNN_ERROR


def _input_val_to_onnx_session_string_option(input_val: Any) -> str:
    """Convert input_value into a string option for an ONNX runtime session."""
    if isinstance(input_val, Enum):
        return str(input_val.value)
    elif isinstance(input_val, bool):
        return "1" if input_val else "0"
    elif isinstance(input_val, Path):
        return str(input_val.resolve().absolute())
    else:
        return str(input_val)


@dataclass()
class OnnxSessionOptions:
    """
    ONNX Runtime session level options.
    """

    enable_mem_pattern: bool = True
    enable_cpu_mem_arena: bool = True
    graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    disable_cpu_ep_fallback: bool = (
        False  # Applies to any execution provider, not just QNN
    )

    ##
    # Options for execution providers that can dump a "session context" (a pre-compiled onnx file) to disk.
    # https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L312
    #
    # NOTE:
    #  A session context is a precompiled ONNX file.
    #  It can be produced by ANY set of execution providers.
    #  "context" != QNN context binary ("context" is an overloaded term here).
    ##

    # We treat `context_enable` differently than standard ONNX behavior.
    # If true, the OnnxModelTorchWrapper API will create the context ONNX file for you and reuse it on future instantiations.
    # This is different than ONNX, which would fail if context_enable is set to True and an ONNX file was created already.
    context_enable: bool = False
    # This is an additional field that ONNX does not provide. If context_enable is also true,
    #   we also hash the input onnx model before writing the cached model to disk (the hash is included in the cache file name).
    # This is optional because the sanity check requires hashing the input ONNX model, which can be expensive.
    #  However, since OnnxModelTorchWrapper is sample code, we enable it by default to reduce potential user pitfalls.
    context_include_onnxfile_hash: bool = True
    context_embed_mode: bool = False
    context_file_path: Path | None = None
    context_node_name_prefix: str | None = None
    share_ep_contexts: bool = False
    stop_share_ep_contexts: bool = False
    # The ONNX documentation for this one is opaque.
    # If this is set, weights for ops that are added directly to the EP-generated context.onnx graph
    # (eg. not in the generated qnn context binary), will be dumped to a separate weights file.
    context_model_external_initializers_file_name: Path | None = None

    @property
    def session_config_fields(self) -> dict[str, str]:
        session_fields = ["disable_cpu_ep_fallback"]
        ep_fields = [
            "context_enable",
            "context_embed_mode",
            "context_file_path",
            "context_node_name_prefix",
            "share_ep_contexts",
            "stop_share_ep_contexts",
            "context_model_external_initializers_file_name",
        ]
        out = {y: f"session.{y}" for y in session_fields}
        out.update({x: f"ep.{x}" for x in ep_fields})
        return out

    @property
    def session_config_entries(self) -> dict[str, str]:
        """
        Convert these options to ONNX runtime session config entries.
        See ExecutionProviderOptions::apply_to_session_options for how to use the returned dict.
        """
        out: dict[str, str] = dict()
        session_config_fields = self.session_config_fields
        for field in fields(self):
            if field.name not in session_config_fields:
                continue
            input_val = getattr(self, field.name)
            if input_val is not None and input_val != field.default:
                out[
                    session_config_fields[field.name]
                ] = _input_val_to_onnx_session_string_option(input_val)

        return out

    @property
    def onnx_session_options(self) -> onnxruntime.SessionOptions:
        """Create ONNX session options from this class."""
        session = onnxruntime.SessionOptions()
        for k, v in self.session_config_entries.items():
            session.add_session_config_entry(k, v)

        session.enable_mem_pattern = self.enable_mem_pattern
        session.enable_cpu_mem_arena = self.enable_cpu_mem_arena
        session.graph_optimization_level = self.graph_optimization_level
        return session

    @classmethod
    def aihub_defaults(cls) -> OnnxSessionOptions:
        options = OnnxSessionOptions()
        options.context_enable = True
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return options

    @property
    def session_context_agnostic_fields(self) -> list[str]:
        """Fields that don't change a session's compiled context."""
        return [
            "context_enable",
            "context_file_path",
            "enable_mem_pattern",
            "enable_cpu_mem_arena",
            "disable_cpu_ep_fallback",
        ]

    def session_context_hash(self, hash: hashlib._Hash | None = None) -> hashlib._Hash:
        """Get a hash that can be used to identify a session context that was compiled with these options applied."""
        return _hash_dataclass(self, self.session_context_agnostic_fields, hash)


@dataclass
class ExecutionProviderOptions:
    """
    Execution provider options base class.
    !!! Default config values should always be equivalent to ONNX defaults !!!
    """

    @classmethod
    def aihub_defaults(cls) -> ExecutionProviderOptions:
        """Get the default settings that AI Hub uses."""
        return cls()

    @property
    @abstractmethod
    def ep_name(self) -> str:
        """The name of the execution provider these options apply to."""
        pass

    @property
    def provider_options_dict(self) -> dict[str, str]:
        """
        Convert these options to an ONNX runtime ep provider options dictionary.

        This dictionary should be passed to
            onnxruntime.InferenceSession(
                ...,
                provider_options=[<this dict>]
            )
        """
        out: dict[str, str] = dict()
        for field in fields(self):
            input_val = getattr(self, field.name)
            if input_val is not None and input_val != field.default:
                out[field.name] = _input_val_to_onnx_session_string_option(input_val)

        return out

    @property
    def session_context_agnostic_fields(self) -> list[str]:
        """Fields that don't change a session's compiled context."""
        return []

    def session_context_hash(self, hash: hashlib._Hash | None = None) -> hashlib._Hash:
        """Get a hash that can be used to identify a session context that was compiled with these options applied."""
        return _hash_dataclass(self, self.session_context_agnostic_fields, hash)


@dataclass
class QNNExecutionProviderOptions(ExecutionProviderOptions):
    """
    Options for the QNN execution provider.
    Default config values are equivalent to ONNX defaults.
    """

    class ProfilingLevel(Enum):
        OFF = "off"
        BASIC = "basic"
        DETAILED = "detailed"

        @classmethod
        def default(cls) -> QNNExecutionProviderOptions.ProfilingLevel:
            return cls.OFF

    class HTPPerformanceMode(Enum):
        BURST = "burst"
        BALANCED = "balanced"
        DEFAULT = "default"
        HIGH_PERFORMANCE = "high_performance"
        HIGH_POWER_SAVER = "high_power_saver"
        LOW_BALANCED = "low_balanced"
        EXTREME_POWER_SAVER = "extreme_power_saver"
        LOW_POWER_SAVER = "low_power_saver"
        POWER_SAVER = "power_saver"
        SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"

        @classmethod
        def default(cls) -> QNNExecutionProviderOptions.HTPPerformanceMode:
            return cls.DEFAULT

    class ContextPriority(Enum):
        LOW = "low"
        NORMAL = "normal"
        NORMAL_HIGH = "normal_high"
        HIGH = "high"

        @classmethod
        def default(cls) -> QNNExecutionProviderOptions.ContextPriority:
            return cls.NORMAL

    class HTPFinalizationOptimizationMode(Enum):
        O1 = 1  # Faster preparation time, less optimal graph.
        O2 = 2  # Longer preparation time, more optimal graph.
        O3 = 3  # Longest preparation time, most likely even more optimal graph. See QNN SDK documentation for specific

        @classmethod
        def default(cls) -> QNNExecutionProviderOptions.HTPFinalizationOptimizationMode:
            return cls.O2

    ####
    #
    # Most options are documented here:
    # https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h#L3753
    #
    ####

    ##
    # Options the typical app developer may want to change.
    ##
    htp_performance_mode: QNNExecutionProviderOptions.HTPPerformanceMode = (
        HTPPerformanceMode.default()
    )
    htp_graph_finalization_optimization_mode: QNNExecutionProviderOptions.HTPFinalizationOptimizationMode = (
        HTPFinalizationOptimizationMode.default()
    )
    qnn_context_priority: QNNExecutionProviderOptions.ContextPriority = (
        ContextPriority.default()
    )

    ##
    # Options typically used for debugging.
    ##
    profiling_level: QNNExecutionProviderOptions.ProfilingLevel = (
        ProfilingLevel.default()
    )
    profiling_file_path: Path | None = None
    dump_json_qnn_graph: bool = False
    json_qnn_graph_dir: Path | None = None
    qnn_saver_path: str | None = None

    ##
    # Advanced options that users rarely need to change.
    ##
    backend_path: Path = Path("QnnHtp.dll")
    rpc_control_latency: int | None = None
    vtcm_mb: int | None = None
    soc_model: int | None = None
    htp_arch: int | None = None
    device_id: int | None = None
    enable_htp_fp16_precision: bool = True
    offload_graph_io_quantization: bool = True
    enable_htp_spill_fill_buffer: bool = False
    enable_htp_shared_memory_allocator: bool = False

    @property
    def ep_name(self) -> str:
        return "QNNExecutionProvider"

    @property
    def provider_options_dict(self) -> dict[str, str]:
        dict = super().provider_options_dict

        # The superclass ignores default values.
        # backend_pack must always be specified
        if "backend_path" not in dict:
            dict["backend_path"] = str(self.backend_path)

        return dict

    @classmethod
    def aihub_defaults(cls) -> QNNExecutionProviderOptions:
        options = QNNExecutionProviderOptions()
        options.htp_performance_mode = (
            QNNExecutionProviderOptions.HTPPerformanceMode.BURST
        )
        options.htp_graph_finalization_optimization_mode = (
            QNNExecutionProviderOptions.HTPFinalizationOptimizationMode.O3
        )
        return options

    @property
    def session_context_agnostic_fields(self) -> list[str]:
        return [
            "profiling_level",
            "profiling_file_path",
            "dump_json_qnn_graph",
            "json_qnn_graph_dir",
            "qnn_saver_path",
            "qnn_context_priority",
            "htp_performance_mode",
        ]


class OnnxSessionTorchWrapper(ExecutableModelProtocol):
    """
    A wrapper for ONNX session that provides a Torch-like inference interface.

    Implements the __call__() and forward() functions in the same way a pyTorch module would.
    This allows this class to act as drop-in replacement for a pyTorch module of the same model.

    The class will also automatically quantize and dequantize model I/O, to make it easier to
    drop the model into floating point-based pipelines even if the ONNX model is quantized.
    """

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        inputs: dict[str, tuple[tuple[int, ...], np.dtype, tuple[float, int] | None]]
        | None = None,
        outputs: dict[str, tuple[tuple[int, ...], np.dtype, tuple[float, int] | None]]
        | None = None,
        quantize_io: bool = True,
    ):
        """
        Create a wrapper for an ONNX session that uses torch-like I/O for the forward call.

        session:
            ONNX session.

        inputs / outputs
            Model inputs / output names & types.
            If not provided, they will be extracted from the session.
            dict[name, tuple[type, (scale, bias)]]

        quantize_io
            If an input is float and the corresponding type in the ONNX model is quantized,
            the input will be automatically quantized for you (when calling __call__ or forward())
            using the QDQ params in the model.

            The same applies to outputs; if an output is quantized, it will be dequantized for you
            (when calling __call__ or forward()) using the QDQ params in the model.

            Set this to false to disable that behavior; instead:
                * if an input type does not match, an error will be raised
                * quantized output will be returned in quantized format
        """
        self.session = session
        self.quantize_io = quantize_io

        if not inputs or not outputs:
            gen_inputs, gen_outputs = extract_io_types_from_onnx_model(session)
            inputs = inputs or gen_inputs
            outputs = outputs or gen_outputs
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Calls the model with the given args and kwargs.
        Identical behavior (I/O) to calling forward() on a pyTorch Module.

        Paramaters:
            *args
                Ordered inputs of any type that can be converted to a numpy array.

            **kwargs
                Keyword inputs of any type that can be converted to a numpy array.

        Returns:
            Model output in default order defined by the ONNX model.
            If the model has 1 output, it will be returned as a Tensor. Otherwise this returns a tuple of Tensors.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Calls the model with the given args and kwargs.
        Identical behavior (I/O) to calling forward() on a pyTorch Module.

        Paramaters:
            *args
                Ordered inputs of any type that can be converted to a numpy array.

            **kwargs
                Keyword inputs of any type that can be converted to a numpy array.

        Returns:
            Model output in default order defined by the ONNX model.
            If the model has 1 output, it will be returned as a Tensor. Otherwise this returns a tuple of Tensors.
        """
        session_inputs = kwargs_to_dict(self.inputs.keys(), *args, **kwargs)
        session_outputs = self.run(session_inputs)
        model_output = [torch.tensor(x) for x in session_outputs]
        return model_output[0] if len(model_output) == 1 else tuple(model_output)

    def run(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        """
        Run the model (equivalent to onnx.InferenceSession.run) with the given inputs.

        Parameters:
            inputs
                Network inputs. Values can be any type that can be converted to a numpy array.

        Returns:
            Network outputs in default order defined by the ONNX model.
        """
        session_inputs = self._prepare_inputs(inputs)
        session_outputs = self.session.run(None, session_inputs)
        return self._process_outputs(session_outputs)

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Prepare the input dictionary by:
            * converting each value to a numpy array
            * casting each value to the associated input type (if applicable)
            * quantizing float values to integers if:
                - qdq parameters are defined in self.inputs
                - self.quantize_io is true

        Parameters:
            inputs
                Network inputs.

        Returns:
            Network inputs compatible with the input dtypes defined by the model.

        Raises:
            ValueError if:
                - "inputs" contains input names that aren't defined by the model.
                - An input's dtype is not compatible with the input dtype defined by the model.
        """
        prepared_inputs: dict[str, np.ndarray] = dict()
        for input_name, input_val in inputs.items():
            if input_name not in self.inputs:
                raise ValueError(
                    f"Unknown input with name {input_name}. Expected inputs: {self.inputs.keys()}"
                )
            _, onnx_input_dtype, qdq_params = self.inputs[input_name]

            if not isinstance(input_val, np.ndarray):
                input_val = np.asarray(input_val)

            if input_val.dtype != onnx_input_dtype:
                input_val_is_float = np.issubdtype(input_val.dtype, np.floating)
                input_val_is_int = not input_val_is_float and np.issubdtype(
                    input_val.dtype, np.integer
                )
                onnx_dtype_is_float = np.issubdtype(onnx_input_dtype, np.floating)
                onnx_dtype_is_int = np.issubdtype(onnx_input_dtype, np.integer)

                if (input_val_is_int and onnx_dtype_is_int) or (
                    input_val_is_float and onnx_dtype_is_float
                ):
                    # Cast the input to the appropriate type if it's the same fundamental type (int / float).
                    input_val = input_val.astype(onnx_input_dtype)
                elif self.quantize_io and input_val_is_float and qdq_params is not None:
                    # Quantize input if it's a float and the target dtype is quantized with known QDQ params.
                    qdq_scale, qdq_bias = qdq_params
                    input_val = ((input_val / qdq_scale) - qdq_bias).astype(
                        onnx_input_dtype
                    )
                else:
                    raise ValueError(
                        f"Input {input_name} has incorrect type {input_val.dtype}. Expected type {onnx_input_dtype}."
                        + (
                            f" If you expected this input to be quantized for you, {self.__class__.__name__} was unable to extract the quantization parameters."
                            if input_val_is_float
                            and onnx_dtype_is_int
                            and self.quantize_io
                            and qdq_params is None
                            else ""
                        )
                    )

            prepared_inputs[input_name] = input_val

        return prepared_inputs

    def _process_outputs(self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Process the output dictionary by:
            * dequantizing integer values to float if:
                - qdq parameters are defined in self.outputs
                - self.quantize_io is true

        Parameters:
            outputs
                Network outputs.

        Returns:
            Processed network outputs.

        Raises:
            ValueError if "outputs" contains a different number of outputs than defined by the ONNX model.
        """
        if len(outputs) != len(self.outputs):
            raise ValueError(
                f"Expected {len(self.outputs)} outputs, but got {len(outputs)} outputs."
            )

        if self.quantize_io:
            processed_outputs: list[np.ndarray] = []
            for idx, (_, _, output_qdq_params) in enumerate(self.outputs.values()):
                output = outputs[idx]
                if output_qdq_params is not None:
                    scale, bias = output_qdq_params
                    output = ((output + bias) * scale).astype(np.float32)
                processed_outputs.append(output)
            return processed_outputs

        return outputs


class OnnxModelTorchWrapper(OnnxSessionTorchWrapper):
    """
    A wrapper for an ONNX model that uses torch-like I/O for the forward call.

    Implements the __call__() and forward() functions in the same way a pyTorch module would.
    This allows this class to act as drop-in replacement for a pyTorch module of the same model.

    The class will also automatically quantize and dequantize model I/O, to make it easier to
    drop the model into floating point-based pipelines even if the ONNX model is quantized.
    """

    def __init__(
        self,
        model_path: str | PathLike,
        session_options: OnnxSessionOptions,
        execution_providers: list[ExecutionProviderOptions],
        quantize_io: bool = True,
    ):
        """
        Create a wrapper for an ONNX model that uses torch-like I/O for the forward call.

        model_path
            ONNX model to load.

        session_options
            ONNX session options. This object will be modified in-place and should not be reused.

        execution_providers
            Execution providers to enable when running this model (& associated settings).

        quantize_io
            If an input is float and the corresponding type in the ONNX model is quantized,
            the input will be automatically quantized for you (when calling __call__ or forward())
            using the QDQ params in the model.

            The same applies to outputs; if an output is quantized, it will be dequantized for you
            (when calling __call__ or forward()) using the QDQ params in the model.

            Set this to false to disable that behavior; instead:
                * if an input type does not match, an error will be raised
                * quantized output will be returned in quantized format
        """
        for ep in execution_providers:
            # Verify the environment is set up correctly for QNN.
            if isinstance(ep, QNNExecutionProviderOptions):
                _verify_onnxruntime_qnn_installed()
                break

        # Extract I/O types & QDQ params from the model.
        # Overwrite / supplement that I/O with user-provided types
        inputs, outputs = extract_io_types_from_onnx_model(
            # External data is not needed here, qdq params are always stored directly in the graph.
            onnx.load(model_path, load_external_data=False)
        )

        # Deal with EP Session Caching
        #   1. Determine location of cache for the user-provided onnx settings.
        #   2. Set up the session options to either load the existing cache or compile a new cache.
        if session_options.context_enable:
            # Hash session options to determine final cache path.
            context_file_path = self._get_model_cache_path(
                model_path, session_options, execution_providers
            )

            # Copy the session options so we can modify it
            session_options = dataclasses.replace(session_options)

            if not os.path.exists(context_file_path):
                # Set the flag that will enable session context compilation if there is no existing context to load.
                session_options.context_file_path = context_file_path
                session_options.context_enable = True
            else:
                # Otherwise, point the inference session to the cached context ONNX model instead of the input model.
                session_options.context_enable = False
                session_options.context_file_path = None
                model_path = context_file_path
                print(f"Loading cached session context at {model_path}")

        # Create the inference session
        self.model_path = model_path
        self.session_options = session_options
        self.execution_providers = execution_providers
        session = onnxruntime.InferenceSession(
            self.model_path,
            self.session_options.onnx_session_options,
            [x.ep_name for x in self.execution_providers],
            [x.provider_options_dict for x in self.execution_providers],
        )

        # A context cache will only be created if:
        #  * session_options.context_enable is True
        #
        #  * one or more selected ONNX EPs can save a session context
        #    (there's no way to know this without checking if the cache was created on disk)
        if (
            session_options.context_enable
            and session_options.context_file_path is not None
            and os.path.exists(session_options.context_file_path)
        ):
            print(f"Saved session context at {session_options.context_file_path}")

        super().__init__(session, inputs, outputs, quantize_io)

    @classmethod
    def OnNPU(
        cls,
        model_path: str | PathLike,
        session_options: OnnxSessionOptions | None = None,
        npu_options: QNNExecutionProviderOptions | None = None,
        quantize_io: bool = True,
    ) -> OnnxModelTorchWrapper:
        """
        Create an executable ONNX model that runs on the Qualcomm NPU via the QNN Execution Provider.

        model_path
            ONNX model to load.

        session_options
            ONNX session options. If undefined, uses AI Hub defaults.
            This object will be modified in-place and should not be reused.

        npu_options
            QNN execution provider options. If undefined, uses AI Hub defaults.

        quantize_io
            If an input is float and the corresponding type in the ONNX model is quantized,
            the input will be automatically quantized for you (when calling __call__ or forward())
            using the QDQ params in the model.

            The same applies to outputs; if an output is quantized, it will be dequantized for you
            (when calling __call__ or forward()) using the QDQ params in the model.

            Set this to false to disable that behavior; instead:
                * if an input type does not match, an error will be raised
                * quantized output will be returned in quantized format
        """
        session_options = session_options or OnnxSessionOptions.aihub_defaults()
        npu_options = npu_options or QNNExecutionProviderOptions.aihub_defaults()
        return cls(
            model_path,
            session_options,
            [npu_options],
            quantize_io,
        )

    @classmethod
    def OnCPU(
        cls,
        model_path: str | PathLike,
        session_options: OnnxSessionOptions | None = None,
        quantize_io: bool = True,
    ):
        """
        Create an executable ONNX model that runs on the CPU.

        model_path
            ONNX model to load.

        session_options
            ONNX session options. If undefined, uses AI Hub defaults.
            This object will be modified in-place and should not be reused.

        quantize_io
            If an input is float and the corresponding type in the ONNX model is quantized,
            the input will be automatically quantized for you (when calling __call__ or forward())
            using the QDQ params in the model.

            The same applies to outputs; if an output is quantized, it will be dequantized for you
            (when calling __call__ or forward()) using the QDQ params in the model.

            Set this to false to disable that behavior; instead:
                * if an input type does not match, an error will be raised
                * quantized output will be returned in quantized format
        """
        session_options = session_options or OnnxSessionOptions.aihub_defaults()
        return cls(
            model_path,
            session_options,
            [],
            quantize_io,
        )

    @classmethod
    def _get_model_cache_path(
        cls,
        model_path: Path | os.PathLike | str,
        session_options: OnnxSessionOptions,
        execution_providers: list[ExecutionProviderOptions],
    ) -> Path:
        # Determine context folder and file name
        if session_options.context_file_path:
            ctx_folder = os.path.dirname(session_options.context_file_path)
            ctx_filename = os.path.splitext(
                os.path.basename(session_options.context_file_path)
            )[0]
        else:
            ctx_folder = os.path.dirname(model_path)
            ctx_filename = f"{os.path.splitext(os.path.basename(model_path))[0]}_ctx"

        # Hash session options to determine final cache path.
        combined_hash = session_options.session_context_hash()
        for ep in execution_providers:
            ep.session_context_hash(combined_hash)
        if session_options.context_include_onnxfile_hash:
            _hash_file(model_path, combined_hash)

        return (
            Path(ctx_folder)
            / f"{ctx_filename}_onnx{onnxruntime.__version__}_{combined_hash.hexdigest()}.onnx"
        )
