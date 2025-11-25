# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Literal

from numpy.typing import ArrayLike, NDArray

from qai_hub_models.utils.runtime_torch_wrapper import RuntimeTorchWrapper
from qai_hub_models.utils.tflite.helpers import (
    Delegate,
    Interpreter,
    TFLiteModelIODetails,
    assert_litert_installed,
    extract_io_types_from_tflite_model,
)


class TFLiteInterpreterTorchWrapper(RuntimeTorchWrapper[TFLiteModelIODetails]):
    """
    A wrapper for TF Lite Interpreter that provides a Torch-like inference interface.

    Implements the __call__() and forward() functions in the same way a pyTorch module would.
    This allows this class to act as drop-in replacement for a pyTorch module of the same model.

    The class will also automatically quantize and dequantize model I/O, to make it easier to
    drop the model into floating point-based pipelines even if the model is quantized.
    """

    def __init__(
        self,
        interpreter: Interpreter,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
    ):
        """
        Create a wrapper for an TF Lite interpreter that uses torch-like I/O for the forward call.

        Parameters
        ----------
        interpreter:
            TFLite model intepreter.

        quantize_user_input
            If a model input is float and the corresponding model input type is
            quantized, the input will be quantized before it is fed to the model.
            Input QDQ params will be extracted from the interpreter.

            - If Sequence[str]: pre-quantization applies only to the input names defined in the sequence
            - If "ALL": pre-quantization applies to all inputs
            - If None: pre-quantization is SKIPPED for all inputs

        dequantize_model_output
            If an output is quantized, the input will be automatically dequantized for you (when calling __call__ or forward())
            Output QDQ params will be extracted from the interpreter.

            - If Sequence[str]: de-quantization applies only to the output names defined in the sequence
            - If "ALL": de-quantization applies to all outputs
            - If None: de-quantization is SKIPPED for all outputs
        """
        self.interpreter = interpreter
        inputs, outputs = extract_io_types_from_tflite_model(interpreter)
        super().__init__(inputs, outputs, quantize_user_input, dequantize_model_output)
        self.interpreter.allocate_tensors()  # required to run the model; no-op if this was run already

    def run(
        self, inputs: Sequence[ArrayLike] | Mapping[str, ArrayLike]
    ) -> list[NDArray]:
        for input_details, value in zip(
            self.inputs.values(), self._prepare_inputs(inputs).values(), strict=False
        ):
            self.interpreter.set_tensor(input_details.graph_tensor_index, value)
        self.interpreter.invoke()
        session_outputs: list[NDArray] = [
            self.interpreter.get_tensor(output_details.graph_tensor_index)
            for output_details in self.outputs.values()
        ]
        return self._process_outputs(session_outputs)


class TFLiteModelTorchWrapper(TFLiteInterpreterTorchWrapper):
    """
    A wrapper for an TF Lite model that uses torch-like I/O for the forward call.

    Implements the __call__() and forward() functions in the same way a pyTorch module would.
    This allows this class to act as drop-in replacement for a pyTorch module of the same model.

    The class will also automatically quantize and dequantize model I/O, to make it easier to
    drop the model into floating point-based pipelines even if the model is quantized.
    """

    def __init__(
        self,
        model_path: str | PathLike,
        delegate_attempt_order: list[list[Delegate]] | None = None,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
        num_threads: int | None = None,
    ):
        """
        Create a wrapper for a TF Lite model that uses torch-like I/O for the forward call.

        Parameters
        ----------
        model_path
            TF Lite model to load.

        delegate_priority_order
            Delegates, in order they should be registered to the interpreter.

            The "inner list" defines which delegates should be registered when creating the interpreter.
            The order of delegates is the priority in which they are assigned layers.
            For example, if an list contains delegates { QNN_NPU, GPUv2 }, then QNN_NPU will be assigned any
            compatible op first. GPUv2 will then be assigned any ops that QNN_NPU is unable to run.
            And finally, XNNPack will be assigned ops that both QNN_NPU and GPUv2 are unable to run.

            The "outer list" defines the order of delegate lists the interpreter should be created with.
            An interpreter will be first created with all delegates in the first list.
            If that interpreter fails to instantiate, an interpreter will be created with all delegates in
            the second list. This continues until an interpreter could be successfully created & returned,
            or until all arrays are tried unsuccessfully--which results in an exception.

            If this is empty or None, defaults to the XNNPack delegate.

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

        num_threads
            Sets the number of threads used by the interpreter and available to CPU kernels.
            If not set, the interpreter will use an implementation-dependent default number of threads.
            If set, num_threads should be >= 1.
        """
        assert_litert_installed()
        interpreter = None
        err: Exception | None = None
        for delegates in delegate_attempt_order or [[]]:
            try:
                interpreter = Interpreter(
                    model_path,
                    experimental_delegates=delegates,
                    num_threads=num_threads,
                )
                self.delegates = delegates
            except ValueError as e:  # noqa: PERF203
                err = e

        if not interpreter:
            raise ValueError(
                f"Unable to create a TF Lite interpreter for this model: {err!s}"
            )

        super().__init__(interpreter, quantize_user_input, dequantize_model_output)

    @classmethod
    def OnCPU(
        cls,
        model_path: str | PathLike,
        quantize_user_input: Sequence[str] | Literal["ALL"] | None = "ALL",
        dequantize_model_output: Sequence[str] | Literal["ALL"] | None = "ALL",
        num_threads: int | None = None,
    ) -> TFLiteModelTorchWrapper:
        """
        Create an executable TF Lite model that runs on the CPU.

        Parameters
        ----------
        model_path
            TF Lite model to load.

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

        num_threads
            Sets the number of threads used by the interpreter and available to CPU kernels.
            If not set, the interpreter will use an implementation-dependent default number of threads.
            If set, num_threads should be >= 1.

        Returns
        -------
        TFLiteModelTorchWrapper
            TF Lite torch wrapper targeting the CPU.
        """
        return TFLiteModelTorchWrapper(
            model_path, None, quantize_user_input, dequantize_model_output, num_threads
        )
