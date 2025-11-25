# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import os
import platform
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import onnx
import onnxruntime
from numpy.typing import ArrayLike, NDArray

from qai_hub_models.utils.asset_loaders import extract_zip_file
from qai_hub_models.utils.onnx.helpers import (
    extract_io_types_from_onnx_model,
    onnx_model_is_precompiled_qairt,
)
from qai_hub_models.utils.runtime_torch_wrapper import (
    ModelIODetails,
    RuntimeTorchWrapper,
)

ONNXRUNTIME_ENV_CHECKED: bool = False
ONNXRUNTIME_QNN_ERROR: ValueError | None = None


def _hash_dataclass(
    cls: object,
    ignore_fields: list[str] | None = None,
    hasher: hashlib._Hash | None = None,
) -> hashlib._Hash:
    hasher = hasher or hashlib.md5()
    for field in dataclasses.fields(cls):  # type: ignore[arg-type]
        if field.name in (ignore_fields or []):
            continue
        hasher.update(
            bytes(f"{field.name}: {getattr(cls, field.name)!s}", encoding="utf-8")
        )
    return hasher


def _hash_file(
    path: Path | os.PathLike | str, hasher: hashlib._Hash | None = None
) -> hashlib._Hash:
    hasher = hasher or hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher


def _verify_onnxruntime_qnn_installed() -> None:
    """
    Throws an exception if onnxruntime-qnn:
        * is not installed
        * has conflicts with other installed packages

    Runs only once then caches the result for this python session.
    """
    global ONNXRUNTIME_ENV_CHECKED  # noqa: PLW0603
    global ONNXRUNTIME_QNN_ERROR  # noqa: PLW0603
    if ONNXRUNTIME_ENV_CHECKED:
        if ONNXRUNTIME_QNN_ERROR:
            raise ONNXRUNTIME_QNN_ERROR
        return

    pkgs = importlib.metadata.distributions()
    pkg_names = {p.name for p in pkgs}

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
                f"    pip uninstall -y {' '.join(additional_runtimes_installed)}{f' {ORT_QNN_PACKAGE_NAME}' if qnn_runtime_installed else ''}",
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
    if isinstance(input_val, bool):
        return "1" if input_val else "0"
    if isinstance(input_val, Path):
        return str(input_val.resolve().absolute())
    return str(input_val)


def extract_onnx_zip(
    path: os.PathLike | str, out_path: Path | None = None, validate_exists: bool = True
) -> tuple[Path, Path]:
    """
    Extract the zip file at the given path and returns the paths
    where the `model.onnx` and `model.data` files can be found.

    Parameters
    ----------
    path
        a folder to validate or zip file to unzip.
        The zip should have been created by AI Hub, or the
        folder should be an unzipped version of a zip
        created by AI Hub.

    out_path
        Folder to which the zip file should be unzipped.
        If None, defaults to the same folder the zip file is in.

    validate
        If True, raises an error if the .onnx file can't be found.

    Returns
    -------
    model_path
        model path (always exists if validate_exists is true)
    model_weights
        model weights path (may not exist, even if validate_exists is true)
    """
    assert os.path.splitext(path)[1].endswith(".zip")
    path = extract_zip_file(path, out_path)

    contents = os.listdir(path=path)
    # Sometimes an extraneous subfolder is created
    onnx_path = path / contents[0] if len(contents) == 1 else path

    model_path = onnx_path / "model.onnx"
    weights_path = onnx_path / "model.data"
    if validate_exists and not os.path.exists(model_path):
        raise ValueError(
            f"model.onnx could not be found at path {model_path}. Was the parent directory created by AI Hub?"
        )

    return (model_path, weights_path)


@dataclass()
class OnnxSessionOptions:
    """ONNX Runtime session level options."""

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
        out: dict[str, str] = {}
        session_config_fields = self.session_config_fields
        for field in fields(self):
            if field.name not in session_config_fields:
                continue
            input_val = getattr(self, field.name)
            if input_val is not None and input_val != field.default:
                out[session_config_fields[field.name]] = (
                    _input_val_to_onnx_session_string_option(input_val)
                )

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

    def session_context_hash(
        self, hasher: hashlib._Hash | None = None
    ) -> hashlib._Hash:
        """Get a hash that can be used to identify a session context that was compiled with these options applied."""
        return _hash_dataclass(self, self.session_context_agnostic_fields, hasher)


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

    @property
    def provider_options_dict(self) -> dict[str, str]:
        """
        Convert these options to an ONNX runtime ep provider options dictionary.

        Returns
        -------
        dict[str, str]
            This dictionary should be passed to
                onnxruntime.InferenceSession(
                    ...,
                    provider_options=[<this dict>]
                )
        """
        out: dict[str, str] = {}
        for field in fields(self):
            input_val = getattr(self, field.name)
            if input_val is not None and input_val != field.default:
                out[field.name] = _input_val_to_onnx_session_string_option(input_val)

        return out

    @property
    def session_context_agnostic_fields(self) -> list[str]:
        """Fields that don't change a session's compiled context."""
        return []

    def session_context_hash(
        self, hasher: hashlib._Hash | None = None
    ) -> hashlib._Hash:
        """Get a hash that can be used to identify a session context that was compiled with these options applied."""
        return _hash_dataclass(self, self.session_context_agnostic_fields, hasher)


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
    htp_graph_finalization_optimization_mode: QNNExecutionProviderOptions.HTPFinalizationOptimizationMode = HTPFinalizationOptimizationMode.default()
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
        provider_options_dict = super().provider_options_dict

        # The superclass ignores default values.
        # backend_pack must always be specified
        if "backend_path" not in provider_options_dict:
            provider_options_dict["backend_path"] = str(self.backend_path)

        return provider_options_dict

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


class OnnxSessionTorchWrapper(RuntimeTorchWrapper[ModelIODetails]):
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
        inputs: dict[str, ModelIODetails] | None = None,
        outputs: dict[str, ModelIODetails] | None = None,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ):
        """
        Create a wrapper for an ONNX session that uses torch-like I/O for the forward call.

        Parameters
        ----------
        session
            ONNX session.

        inputs
            Ordered model input names & types.
            **dict entry order match the input declaration order in the model.**

            If not provided, inputs will be extracted from the session.
            Note that quantization parameters cannot be extracted from an ONNX session.

        outputs
            Ordered model output names & types.
            **dict entry order match the output declaration order in the model.**

            If not provided, outputs will be extracted from the session.
            Note that quantization parameters cannot be extracted from an ONNX session.

        quantize_user_input
            If a model input is float and the corresponding model input type is
            quantized, the input will be quantized before it is fed to the model.
            The QDQ params specified in `inputs` will be used for quantization.

            - If Sequence[str]: pre-quantization applies only to the input names defined in the sequence
            - If "ALL": pre-quantization applies to all inputs
            - If None: pre-quantization is SKIPPED for all inputs

        dequantize_model_output
            If an output is quantized, the input will be automatically dequantized for you (when calling __call__ or forward())
            The QDQ params specified in `outputs` will be used for quantization.

            - If Sequence[str]: de-quantization applies only to the output names defined in the sequence
            - If "ALL": de-quantization applies to all outputs
            - If None: de-quantization is SKIPPED for all outputs
        """
        self.session = session
        if not inputs or not outputs:
            gen_inputs, gen_outputs = extract_io_types_from_onnx_model(session)
            inputs = inputs or gen_inputs
            outputs = outputs or gen_outputs
        super().__init__(inputs, outputs, quantize_user_input, dequantize_model_output)

    def run(
        self, inputs: Sequence[ArrayLike] | Mapping[str, ArrayLike]
    ) -> list[NDArray]:
        session_inputs = self._prepare_inputs(inputs)
        session_outputs = cast(
            Sequence[NDArray], self.session.run(None, session_inputs)
        )
        return self._process_outputs(session_outputs)


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
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ):
        """
        Create a wrapper for an ONNX model that uses torch-like I/O for the forward call.

        Parameters
        ----------
        model_path
            ONNX model to load.

        session_options
            ONNX session options. This object will be modified in-place and should not be reused.

        execution_providers
            Execution providers to enable when running this model (and associated settings).

        quantize_user_input
            If a model input is float and the corresponding model input type is
            quantized, the input will be quantized before it is fed to the model.
            Input QDQ params will be extracted from the model file.

            - If Sequence[str]: pre-quantization applies only to the input names defined in the sequence
            - If "ALL": pre-quantization applies to all inputs
            - If None: pre-quantization is SKIPPED for all inputs

        dequantize_model_output
            If an output is quantized, the input will be automatically dequantized for you (when calling __call__ or forward())
            Output QDQ params will be extracted from the model file.

            - If Sequence[str]: de-quantization applies only to the output names defined in the sequence
            - If "ALL": de-quantization applies to all outputs
            - If None: de-quantization is SKIPPED for all outputs
        """
        for ep in execution_providers:
            # Verify the environment is set up correctly for QNN.
            if isinstance(ep, QNNExecutionProviderOptions):
                _verify_onnxruntime_qnn_installed()
                break

        if str(model_path).endswith(".zip"):
            model_path = extract_onnx_zip(str(model_path))[0]

        onnx_model = onnx.load(model_path, load_external_data=False)

        # Extract I/O types & QDQ params from the model.
        # Overwrite / supplement that I/O with user-provided types
        # External data is not needed here, qdq params are always stored directly in the graph.
        inputs, outputs = extract_io_types_from_onnx_model(onnx_model)

        # Deal with EP Session Caching
        #   1. Determine location of cache for the user-provided onnx settings.
        #   2. Set up the session options to either load the existing cache or compile a new cache.
        if session_options.context_enable:
            # Copy the session options so we can modify it
            session_options = dataclasses.replace(session_options)

            # TODO: Remove onnx_model_is_precompiled_qairt after we update from ONNX 1.22.1.
            #       This is needed because of a bug in ORT that causes precompiled models to fail if this session option is on.
            if onnx_model_is_precompiled_qairt(onnx_model):
                session_options.context_enable = False
            else:
                # Hash session options to determine final cache path.
                context_file_path = self._get_model_cache_path(
                    model_path, session_options, execution_providers
                )

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

        super().__init__(
            session, inputs, outputs, quantize_user_input, dequantize_model_output
        )

    @classmethod
    def OnNPU(
        cls,
        model_path: str | PathLike,
        session_options: OnnxSessionOptions | None = None,
        npu_options: QNNExecutionProviderOptions | None = None,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ) -> OnnxModelTorchWrapper:
        """
        Create an executable ONNX model that runs on the Qualcomm NPU via the QNN Execution Provider.

        Parameters
        ----------
        model_path
            ONNX model to load.

        session_options
            ONNX session options. If undefined, uses AI Hub defaults.
            This object will be modified in-place and should not be reused.

        npu_options
            QNN execution provider options. If undefined, uses AI Hub defaults.

        quantize_user_input
            If a model input is float and the corresponding model input type is
            quantized, the input will be quantized before it is fed to the model.
            Input QDQ params will be extracted from the model file.

            - If Sequence[str]: pre-quantization applies only to the input names defined in the sequence
            - If "ALL": pre-quantization applies to all inputs
            - If None: pre-quantization is SKIPPED for all inputs

        dequantize_model_output
            If an output is quantized, the input will be automatically dequantized for you (when calling __call__ or forward())
            Output QDQ params will be extracted from the model file.

            - If Sequence[str]: de-quantization applies only to the output names defined in the sequence
            - If "ALL": de-quantization applies to all outputs
            - If None: de-quantization is SKIPPED for all outputs

        Returns
        -------
        OnnxModelTorchWrapper
            Wrapped torch model that runs on the Qualcomm NPU via the QNN Execution Provider.
        """
        session_options = session_options or OnnxSessionOptions.aihub_defaults()
        npu_options = npu_options or QNNExecutionProviderOptions.aihub_defaults()
        return cls(
            model_path,
            session_options,
            [npu_options],
            quantize_user_input,
            dequantize_model_output,
        )

    @classmethod
    def OnCPU(
        cls,
        model_path: str | PathLike,
        session_options: OnnxSessionOptions | None = None,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ):
        """
        Create an executable ONNX model that runs on the CPU.

        Parameters
        ----------
        model_path
            ONNX model to load.

        session_options
            ONNX session options. If undefined, uses AI Hub defaults.
            This object will be modified in-place and should not be reused.

        quantize_user_input
            If a model input is float and the corresponding model input type is
            quantized, the input will be quantized before it is fed to the model.
            Input QDQ params will be extracted from the model file.

            - If Sequence[str]: pre-quantization applies only to the input names defined in the sequence
            - If "ALL": pre-quantization applies to all inputs
            - If None: pre-quantization is SKIPPED for all inputs

        dequantize_model_output
            If an output is quantized, the input will be automatically dequantized for you (when calling __call__ or forward())
            Output QDQ params will be extracted from the model file.

            - If Sequence[str]: de-quantization applies only to the output names defined in the sequence
            - If "ALL": de-quantization applies to all outputs
            - If None: de-quantization is SKIPPED for all outputs

        Returns
        -------
        OnnxModelTorchWrapper
            Wrapped torch model that runs on the CPU.
        """
        session_options = session_options or OnnxSessionOptions.aihub_defaults()
        return cls(
            model_path,
            session_options,
            [],
            quantize_user_input,
            dequantize_model_output,
        )

    @classmethod
    def _get_model_cache_path(
        cls,
        model_path: Path | os.PathLike | str,
        session_options: OnnxSessionOptions,
        execution_providers: list[ExecutionProviderOptions],
    ) -> Path:
        """When models are compiled on-device, ONNXModelTorchWrapper can cache the compiled model to speed up subsequent loads. This gets the location on disk where the cached asset should live."""
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
