# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional

import qai_hub
import torch
from qai_hub.client import Device, SourceModel

from qai_hub_models.models.common import (
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.protocols import (
    ExecutableModelProtocol,
    FromPrecompiledProtocol,
    HubModelProtocol,
    PretrainedHubModelProtocol,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs


class CollectionModel:
    """
    Model that glues together several BaseModels
    """

    pass


class HubModel(HubModelProtocol):
    """
    Base interface for AI Hub models.
    """

    def __init__(self):
        # If a child class implements _get_input_spec_for_instance(),
        # then calling `get_input_spec` on the instance will redirect to it.
        if self._get_input_spec_for_instance.__module__ != __name__:
            self.get_input_spec = self._get_input_spec_for_instance
        if self._get_output_names_for_instance.__module__ != __name__:
            self.get_output_names = self._get_output_names_for_instance

    def _get_input_spec_for_instance(self, *args, **kwargs) -> InputSpec:
        """
        Get the input specifications for an instance of this model.

        Typically this will pre-fill inputs of get_input_spec
        with values determined by instance members of the model class.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_input_spec with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_output_names_for_instance(self, *args, **kwargs) -> List[str]:
        """
        Get the output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_output_names with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        """
        Returns a set of sample inputs for the model.

        For each input name in the model, a list of numpy arrays is provided.
        If the returned set is batch N, all input names must contain exactly N numpy arrays.

        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs_dict = {}
        inputs_list = make_torch_inputs(input_spec)
        for i, input_name in enumerate(input_spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        return inputs_dict

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        """
        AI Hub profile options recommended for the model.
        """
        return other_profile_options


class BaseModel(
    torch.nn.Module,
    HubModel,
    PretrainedHubModelProtocol,
    ExecutableModelProtocol,
):
    """
    A pre-trained PyTorch model with helpers for submission to AI Hub.
    """

    def __init__(self):
        torch.nn.Module.__init__(self)  # Initialize Torch Module
        HubModel.__init__(self)  # Initialize Hub Model
        self.eval()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
            If this model is being trained, calling `model.train()`
            will reverse all of these.
        """
        if isinstance(value, torch.nn.Module) and not self.training:
            value.eval()
        torch.nn.Module.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        """
        If a model is in eval mode (which equates to self.training == False),
            we don't want to compute gradients when doing the forward pass.
        """
        context_fn = nullcontext if self.training else torch.no_grad
        with context_fn():
            return torch.nn.Module.__call__(self, *args, **kwargs)

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        if not input_spec:
            input_spec = self.get_input_spec()

        # Torchscript should never be trained, so disable gradients for all parameters.
        # Need to do this on a model copy, in case the original model is being trained.
        model_copy = deepcopy(self)
        for param in model_copy.parameters():
            param.requires_grad = False

        return torch.jit.trace(
            model_copy, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: Optional[List[str]] = None,
    ) -> SourceModel:
        """
        Convert to a AI Hub source model appropriate for the export method.
        """
        # Local import to prevent circular dependency
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        source_model, _ = prepare_compile_zoo_model_to_hub(
            self,
            source_model_format=self.preferred_hub_source_model_format(target_runtime),
            target_runtime=target_runtime,
            output_path=output_path,
            input_spec=input_spec,
            check_trace=check_trace,
            external_onnx_weights=external_onnx_weights,
            output_names=output_names,
        )
        return source_model

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        """
        AI Hub compile options recommended for the model.
        """
        target_runtime_flag = None
        if "--target_runtime" not in other_compile_options:
            if target_runtime == TargetRuntime.QNN:
                if device:
                    if not device.attributes:
                        # Only name / os specified
                        devices = qai_hub.get_devices(device.name, device.os)
                    elif not device.name:
                        # Only attribute specified
                        devices = qai_hub.get_devices(attributes=device.attributes)
                    else:
                        devices = [device]

                    for device in devices:
                        if (
                            "os:android" not in device.attributes
                            or "format:iot" in device.attributes
                        ):
                            target_runtime_flag = "qnn_context_binary"
                            break

                target_runtime_flag = target_runtime_flag or "qnn_lib_aarch64_android"
            elif target_runtime == TargetRuntime.ONNX:
                target_runtime_flag = "onnx"
            elif target_runtime == TargetRuntime.TFLITE:
                target_runtime_flag = "tflite"
            elif target_runtime == TargetRuntime.PRECOMPILED_QNN_ONNX:
                target_runtime_flag = "precompiled_qnn_onnx"
            else:
                raise NotImplementedError()

        compile_options = (
            f"--target_runtime {target_runtime_flag}" if target_runtime_flag else ""
        )
        compile_options += f" --output_names {','.join(self.get_output_names())}"

        if other_compile_options != "":
            return compile_options + " " + other_compile_options

        return compile_options

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.TORCHSCRIPT


class BasePrecompiledModel(HubModel, FromPrecompiledProtocol):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str):
        self.target_model_path = target_model_path

    def get_target_model_path(self) -> str:
        """Get the path to the compiled asset for this model on disk."""
        return self.target_model_path
