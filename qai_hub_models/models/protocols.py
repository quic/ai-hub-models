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
from typing import Protocol, Type, TypeVar, runtime_checkable

from qai_hub.client import DatasetEntries

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, _DataLoader
from qai_hub_models.models.common import SampleInputsType, TargetRuntime
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


class QuantizableModelProtocol(Protocol):
    """
    Methods required for a model to be quantizable.
    """

    @abstractmethod
    def quantize(
        self,
        data: _DataLoader,
        num_samples: int | None = None,
        evaluator: BaseEvaluator | None = None,
        device: str = "cpu",
        requantize_model_weights=False,
    ) -> float | None:
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

            evaluator: BaseModelEvaluator | None
                Evaluator to populate while quantizing the data.
                If not provided, an evaluator is not used.

            device: str
                Name of device on which inference should be run.

            requantize_model_weights: bool
                If a weight is quantized, recompute its quantization parameters.

        Returns:
            If an evaluator is provided, returns its accuracy score. No return value otherwise.
        """
        ...

    @abstractmethod
    def get_calibration_data(
        self,
        target_runtime: TargetRuntime,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model and input spec.
        """
        ...


class ExecutableModelProtocol(Protocol):
    """
    Classes follow this protocol if they are executable.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
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
        cls: Type[FromPretrainedTypeVar], *args, **kwargs
    ) -> FromPretrainedTypeVar:
        """
        Utility function that helps users get up and running with a default
        pretrained model. While this function may take arguments, all arguments
        should have default values specified, so that all classes can be invoked
        with `cls.from_pretrained()` and always have it return something reasonable.
        """
        ...


class FromPrecompiledProtocol(Protocol):
    """
    Models follow this protocol if they can be initiated from a precompiled torch model.
    """

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
        ...
