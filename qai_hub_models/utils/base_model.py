# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from inspect import getmodule
from typing import Any, Dict, List, Type, TypeVar

import numpy as np
import torch
from qai_hub.client import SourceModel

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs

InputsType = Dict[str, List[np.ndarray]]


class TargetRuntime(Enum):
    TFLITE = 0
    QNN = 1

    def __str__(self):
        return self.name.lower()


class SourceModelFormat(Enum):
    ONNX = 0
    TORCHSCRIPT = 1


class DocstringInheritorMeta(ABCMeta):
    """
    Ensures that all subclasses retain the `forward` function's docstring.
    """

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        if hasattr(new_class, "forward"):
            parent_method = getattr(bases[0], "forward", None)
            if parent_method and new_class.forward.__doc__ is None:  # type: ignore
                new_class.forward.__doc__ = parent_method.__doc__  # type: ignore
        return new_class


# Use this for typehints that take in a class and output an instance of the class.
FromPretrainedTypeVar = TypeVar("FromPretrainedTypeVar", bound="FromPretrainedMixin")
FromPrecompiledTypeVar = TypeVar("FromPrecompiledTypeVar", bound="FromPrecompiledMixin")


class FromPretrainedMixin(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(
        cls: Type[FromPretrainedTypeVar], *args, **kwargs
    ) -> FromPretrainedTypeVar:
        """
        Utility function that helps users get up and running with a default
        pretrained model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_pretrained()` and always have it return something reasonable.
        """
        pass


class CollectionModel(FromPretrainedMixin):
    """
    Model that glues together several BaseModels
    """

    pass


class BaseModel(
    torch.nn.Module, FromPretrainedMixin, ABC, metaclass=DocstringInheritorMeta
):
    @abstractmethod
    def get_input_spec(self, *args, **kwargs) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        pass

    @classmethod
    def get_model_id(cls) -> str:
        """
        Return model ID for this model.
        The model ID is the same as the folder name for the model under qai_hub_models/models/...
        """
        module = getmodule(cls)
        if not module or not module.__file__:
            raise ValueError(f"Unable to get model ID for {cls.__name__}")

        # Module path is always .../qai_hub_models/models/<model_id>/model.py
        # Extract model ID from that path.
        return os.path.basename(os.path.dirname(module.__file__))

    def get_evaluator(self) -> BaseEvaluator:
        """
        Gets default model output evaluator for this model.
        """
        raise NotImplementedError("This model does not define a default evaluator.")

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
        output_path: str,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
    ) -> SourceModel:
        """
        Convert to a AI Hub source model appropriate for the export method.
        """
        # Local import to prevent circular dependency
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        assert isinstance(self, BaseModel)
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
        Convert to a AI Hub source model appropriate for the export method.
        """
        compile_options = ""
        if target_runtime == TargetRuntime.QNN:
            compile_options = "--target_runtime qnn_lib_aarch64_android"
        if other_compile_options != "":
            return compile_options + " " + other_compile_options
        return compile_options

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.TORCHSCRIPT

    def sample_inputs(self, input_spec: InputSpec | None = None) -> InputsType:
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


class FromPrecompiledMixin(ABC):
    @classmethod
    @abstractmethod
    def from_precompiled(
        cls: Type[FromPrecompiledTypeVar], *args, **kwargs
    ) -> "FromPrecompiledTypeVar":
        """
        Utility function that helps users get up and running with a default
        precompiled model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_precompiled()` and always have it return something reasonable.
        """
        pass


class BasePrecompiledModel(FromPrecompiledMixin):
    @abstractmethod
    def get_input_spec(self, *args, **kwargs) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        pass

    def sample_inputs(self, input_spec: InputSpec | None = None) -> InputsType:
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
