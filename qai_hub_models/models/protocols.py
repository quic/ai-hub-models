# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This file defines type helpers. Specifically, those helpers are python Protocols.

Protocols are helpful for defining interfaces that must be implemented for specific functions.

For example, a function may take any class that implements FromPretrained.
The parameter would be typed "FromPretrainedProtocol", as defined in this file.

Protocols may also be inherited to declare that a class must implement said protocol.
For example, AIMETQuantizableMixin inherits HubModelProtocol. This informs the type
checker that the class that inherits the mixin must implement HubModelProtocol.

These are type checked at compile time.
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, TypeVar, runtime_checkable

from qai_hub.client import DatasetEntries, Device, SourceModel

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, _DataLoader
from qai_hub_models.models.common import (
    Precision,
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

FromPretrainedTypeVar = TypeVar("FromPretrainedTypeVar", bound="FromPretrainedProtocol")
FromPrecompiledTypeVar = TypeVar(
    "FromPrecompiledTypeVar", bound="FromPrecompiledProtocol"
)


class HubModelProtocol(Protocol):
    """
    All AI Hub Models must, at minimum, implement this interface.
    """

    @staticmethod
    @abstractmethod
    def get_input_spec(*args, **kwargs) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        ...

    @abstractmethod
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
        ...

    @staticmethod
    @abstractmethod
    def get_output_names(*args, **kwargs) -> list[str]:
        """
        List of output names. If there are multiple outputs, the order of the names
            should match the order of tuple returned by the model.
        """
        ...


class QuantizableModelProtocol(Protocol):
    """
    Methods required for a model to be quantizable.
    """

    @abstractmethod
    def quantize(
        self,
        data: _DataLoader,
        num_samples: int | None = None,
        device: str = "cpu",
        requantize_model_weights=False,
        data_has_gt=False,
    ) -> None:
        """
        Compute quantization encodings for this model with the given dataset and model evaluator.

        This model will be updated with a new set of quantization parameters. Future calls to
        forward() and export_...() will take these quantization parameters into account.

        Parameters:
            data: torch DataLoader | Collection
                Data loader for the dataset to use for evaluation.
                    If an evaluator is __NOT__ provided (see "evaluator" parameter), the iterator must return
                        inputs: Collection[torch.Tensor] | torch.Tensor

                    otherwise, if an evaluator __IS__ provided, the iterator must return
                        tuple(
                          inputs: Collection[torch.Tensor] | torch.Tensor,
                          ground_truth: Collection[torch.Tensor] | torch.Tensor]
                        )

            num_samples: int | None
                Number of samples to use for evaluation. One sample is one iteration from iter(data).
                If none, defaults to the number of samples in the dataset.

            device: str
                Name of device on which inference should be run.

            requantize_model_weights: bool
                If a weight is quantized, recompute its quantization parameters.

            data_has_gt: bool
                Set to true if the data loader passed in also provides ground truth data.
                The ground truth data will be discarded for quantization.
        """
        ...

    @abstractmethod
    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model and input spec.
        """
        ...


T = TypeVar("T", covariant=True)


class ExecutableModelProtocol(Generic[T], Protocol):
    """
    Classes follow this protocol if they are executable.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> T:
        """
        Execute the model and return its output.
        """
        ...


@runtime_checkable
class EvalModelProtocol(Protocol):
    """
    Models follow this protocol if they can be numerically evaluated.
    """

    @abstractmethod
    def get_evaluator(self) -> BaseEvaluator:
        """
        Gets a class for evaluating output of this model.
        """
        ...


@runtime_checkable
class FromPretrainedProtocol(Protocol):
    """
    Models follow this protocol if they can be initiated from a pretrained torch model.
    """

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls: type[FromPretrainedTypeVar], *args, **kwargs
    ) -> FromPretrainedTypeVar:
        """
        Utility function that helps users get up and running with a default
        pretrained model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_pretrained()` and always have it return something reasonable.
        """
        ...


class PretrainedHubModelProtocol(HubModelProtocol, FromPretrainedProtocol, Protocol):
    """
    All pretrained AI Hub Models must, at minimum, implement this interface.
    """

    @abstractmethod
    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        ...

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: Optional[list[str]] = None,
    ) -> SourceModel:
        ...

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        """
        AI Hub compile options recommended for the model.
        """
        ...

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        ...

    def get_hub_quantize_options(self, precision: Precision) -> str:
        """
        AI Hub quantize options recommended for the model.
        """
        ...


class FromPrecompiledProtocol(Protocol):
    """
    Models follow this protocol if they can be initiated from a precompiled torch model.
    """

    @classmethod
    @abstractmethod
    def from_precompiled(
        cls: type[FromPrecompiledTypeVar], *args, **kwargs
    ) -> FromPrecompiledTypeVar:
        """
        Utility function that helps users get up and running with a default
        precompiled model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_precompiled()` and always have it return something reasonable.
        """
        ...
