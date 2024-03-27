# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from qai_hub.client import SourceModel

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
        # Change self.get_input_spec() to call _get_input_spec_for_instance() instead.
        #
        # _get_input_spec_for_instance() is an override that allows get_input_spec()
        # to access instance variables. This may be used in case input shape is "hard-coded"
        # based on parameters passed to the model upon initialization.
        #
        self.get_input_spec = self._get_input_spec_for_instance

    def _get_input_spec_for_instance(self, *args, **kwargs) -> InputSpec:
        """
        Get the input specifications for an instance of this model.

        Typically this will pre-fill inputs of get_input_spec
        with values determined by instance members of the model class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(*args, **kwargs)

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

        return torch.jit.trace(
            self, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
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
        )
        return source_model

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
    ) -> str:
        """
        AI Hub compile options recommended for the model.
        """
        compile_options = ""
        if target_runtime == TargetRuntime.QNN:
            compile_options = "--target_runtime qnn_lib_aarch64_android"
        if target_runtime == TargetRuntime.ORT:
            compile_options = "--target_runtime onnx"
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
