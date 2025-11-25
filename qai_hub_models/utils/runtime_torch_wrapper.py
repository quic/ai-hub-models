# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from qai_hub_models.models.protocols import ExecutableModelProtocol


@dataclass
class ModelIODetails:
    shape: tuple[int, ...]
    dtype: np.dtype

    @dataclass
    class QDQParams:
        # Quantize: (float / scale) + zero_point
        # Dequantize: (qint - zero_point) * scale
        scale: float
        zero_point: int

    qdq_params: QDQParams | None


ModelIODetailsT = TypeVar("ModelIODetailsT", bound=ModelIODetails)


class RuntimeTorchWrapper(ABC, ExecutableModelProtocol, Generic[ModelIODetailsT]):
    """
    A wrapper for an on-device runtime that provides a Torch-like inference interface.

    Implements the __call__() and forward() functions in the same way a pyTorch module would.
    This allows this class to act as drop-in replacement for a pyTorch module of the same model.

    The class will also automatically quantize and dequantize model I/O, to make it easier to
    drop the model into floating point-based pipelines even if model I/O is quantized.
    """

    def __init__(
        self,
        inputs: dict[str, ModelIODetailsT],
        outputs: dict[str, ModelIODetailsT],
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ):
        """
        Parameters
        ----------
        inputs
            Model input names & types.
            **dict entry order match the input declaration order in the model.**

        outputs
            Model output names & types.
            **dict entry order match the output declaration order in the model.**

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
        self.inputs = inputs
        self.outputs = outputs

        self.quantize_user_input: set[str] = set()
        if quantize_user_input is not None:
            if quantize_user_input == "ALL":
                self.quantize_user_input = set(self.inputs.keys())
            else:
                self.quantize_user_input = set(quantize_user_input)

        self.dequantize_model_output: set[str] = set()
        if dequantize_model_output is not None:
            if dequantize_model_output == "ALL":
                self.dequantize_model_output = set(self.outputs.keys())
            else:
                self.dequantize_model_output = set(dequantize_model_output)

    def __call__(
        self, *args: ArrayLike, **kwargs: ArrayLike
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Calls the model with the given args and kwargs.
        Identical behavior (I/O) to calling forward() on a pyTorch Module.

        Paramaters
        ----------
        *args
            Ordered model inputs of any type that can be converted to a numpy array.
            Must be in the same order as self.inputs.

        **kwargs
            Keyword model inputs of any type that can be converted to a numpy array.
            Order is not considered.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Model output in the same order as elements of self.outputs.
            If the model has 1 output, it will be returned as a Tensor. Otherwise this returns a tuple of Tensors.
        """
        return self.forward(*args, **kwargs)

    def forward(
        self, *args: ArrayLike, **kwargs: ArrayLike
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Calls the model with the given args and kwargs.
        Identical behavior (I/O) to calling forward() on a pyTorch Module.

        Paramaters
        ----------
        *args
            Ordered model inputs of any type that can be converted to a numpy array.
            Must be in the same order as self.inputs.

        **kwargs
            Keyword model inputs of any type that can be converted to a numpy array.
            Order is not considered.

        Returns
        -------
            Model output in the same order as elements of self.outputs.
            If the model has 1 output, it will be returned as a Tensor. Otherwise this returns a tuple of Tensors.
        """
        session_inputs = kwargs_to_dict(self.inputs.keys(), *args, **kwargs)
        session_outputs = self.run(session_inputs)
        model_output = [
            torch.from_numpy(x) for x in session_outputs
        ]  # from_numpy() creates a tensor that shares memory with the np array
        return model_output[0] if len(model_output) == 1 else tuple(model_output)

    @abstractmethod
    def run(
        self, inputs: Sequence[ArrayLike] | Mapping[str, ArrayLike]
    ) -> list[NDArray]:
        """
        Run the model with the given inputs.

        Parameters
        ----------
        inputs
            Model inputs. Values can be any type that can be converted to a numpy array.
            If this is a sequence, order must match the order of self.inputs.
            If this is a mapping, keys must match those defined by self.inputs, and order is not considered.

        Returns
        -------
        list[NDArray]
            Model output in the same order as elements of self.outputs.
        """

    def _prepare_inputs(
        self, inputs: Sequence[ArrayLike] | Mapping[str, ArrayLike]
    ) -> dict[str, NDArray]:
        """
        Prepare the input dictionary by:
            * converting each value to a numpy array
            * casting each value to the associated input type (if applicable)
            * quantizing float values to integers if:
                - qdq parameters are defined in self.inputs
                - self.quantize_user_input is true

        Parameters
        ----------
        inputs
            Model inputs
            If this is a sequence, order must match the order of self.inputs.
            If this is a mapping, keys must match those defined by self.inputs, and order is not considered.

        Returns
        -------
        dict[str, NDArray]
            Processed nodel inputs, in order of self.inputs.

        Raises
        ------
        ValueError
            If:
            - "inputs" contains input names that aren't defined by the model.
            - An input's dtype is not compatible with the input dtype defined by the model.
        """
        if isinstance(inputs, Mapping):
            if inputs.keys() != self.inputs.keys():
                raise ValueError(
                    f"Inputs are not compatible with the model. Expected ( {self.inputs.keys()}), but got ({inputs.keys()})"
                )
            inputs = [inputs[iname] for iname in self.inputs]

        prepared_inputs: dict[str, NDArray] = {}
        for (input_name, input_details), input_val in zip(
            self.inputs.items(), inputs, strict=False
        ):
            if input_name not in self.inputs:
                raise ValueError(
                    f"Unknown input with name {input_name}. Expected inputs: {self.inputs.keys()}"
                )
            input_details = self.inputs[input_name]

            if isinstance(input_val, torch.Tensor):
                # tensor.numpy() creates a np array that shares memory with the torch tensor
                input_val = input_val.numpy()
            else:
                input_val = np.asarray(input_val)

            if input_val.dtype != input_details.dtype:
                input_val_is_float = np.issubdtype(input_val.dtype, np.floating)
                input_val_is_int = not input_val_is_float and np.issubdtype(
                    input_val.dtype, np.integer
                )
                model_dtype_is_float = np.issubdtype(input_details.dtype, np.floating)
                model_dtype_is_int = np.issubdtype(input_details.dtype, np.integer)

                if (
                    (input_val_is_int and model_dtype_is_int)
                    or (input_val_is_float and model_dtype_is_float)
                    or (
                        input_details.dtype.itemsize >= 4
                        and input_details.qdq_params is None
                    )
                ):
                    # Cast the input to the appropriate type if either:
                    #  - it's the same fundamental type (int / float)
                    #   or
                    #  - the destination type is 32-bit or greater.
                    input_val = input_val.astype(input_details.dtype)
                elif (
                    self.quantize_user_input
                    and input_name in self.quantize_user_input
                    and input_val_is_float
                    and input_details.qdq_params is not None
                ):
                    # Quantize input if it's a float and the target dtype is quantized with known QDQ params.
                    input_val = (
                        np.rint(input_val / input_details.qdq_params.scale)
                    ).astype(input_details.dtype) + input_details.qdq_params.zero_point
                else:
                    raise ValueError(
                        f"Input {input_name} has incorrect type {input_val.dtype}. Expected type {input_details.dtype}."
                        + (
                            f" If you expected this input to be quantized for you, {self.__class__.__name__} was unable to extract the quantization parameters."
                            if input_val_is_float
                            and model_dtype_is_int
                            and input_name in self.quantize_user_input
                            and input_details.qdq_params is None
                            else ""
                        )
                    )

            prepared_inputs[input_name] = input_val

        return prepared_inputs

    def _process_outputs(self, outputs: Sequence[NDArray]) -> list[NDArray]:
        """
        Process the output dictionary by:
            * dequantizing integer values to float if:
                - qdq parameters are defined in self.outputs
                - self.dequantize_model_output is true

        Parameters
        ----------
        outputs
            Model outputs, in order of self.outputs.

        Returns
        -------
        list[NDArray]
            Processed model outputs, in order of self.outputs

        Raises
        ------
        ValueError
            If "outputs" contains a different number of outputs than defined by the model.
        """
        if len(outputs) != len(self.outputs):
            raise ValueError(
                f"Expected {len(self.outputs)} outputs, but got {len(outputs)} outputs."
            )

        if self.dequantize_model_output:
            processed_outputs: list[NDArray] = []
            for output, (output_name, output_details) in zip(
                outputs, self.outputs.items(), strict=False
            ):
                if (
                    output_details.qdq_params is not None
                    and output_name in self.dequantize_model_output
                ):
                    output = (
                        output - np.int32(output_details.qdq_params.zero_point)
                    ) * np.float32(output_details.qdq_params.scale)
                processed_outputs.append(output)
            return processed_outputs

        return list(outputs)


T = TypeVar("T")


def kwargs_to_dict(argnames: Iterable[str], *args: T, **kwargs: T) -> dict[str, T]:
    """
    Convert args + kwargs to a key / value dictionary.

    Parameters
    ----------
    argnames
        Argument names, in order. Orderd arguments will be mapped to these names.

    args
        Ordered arguments

    kwargs
        Keyword arguments

    Returns
    -------
    dict[str, T]
        Ordered key / value dictionary, in order of "argnames"

    Raises
    ------
    ValueError
        if an input is passed twice or an argname is missing.
    """
    input_dict: dict[str, T] = {}
    for idx, input_name in enumerate(argnames):
        if len(args) > idx:
            input_val = args[idx]
            if input_name in kwargs:
                raise ValueError(
                    f"Cannot pass input {input_name} twice (as a positional arg and a keyword arg)."
                )
        elif input_name in kwargs:
            input_val = kwargs[input_name]
        else:
            raise ValueError(f"Missing input {input_name}")
        input_dict[input_name] = input_val
    return input_dict
