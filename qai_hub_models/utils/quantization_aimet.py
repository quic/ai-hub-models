# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

try:
    from aimet_torch import onnx_utils
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    from aimet_torch.quantsim import QuantizationSimModel
except (ImportError, ModuleNotFoundError):
    raise NotImplementedError(
        "AIMET must be installed to load quantized models. "
        "AIMET is only supported on Linux. "
        "Install AIMET via the instructions here: "
        "https://quic.github.io/aimet-pages/releases/latest/install/index.html"
    )

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.evaluators.base_evaluators import (
    BaseEvaluator,
    _DataLoader,
    _for_each_batch,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    InputSpec,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import make_torch_inputs


class AIMETQuantizableMixin:
    """
    This mixin provides quantization support with Qualcomm's AIMET package.
    """

    def __init__(
        self,
        sim_model: QuantizationSimModel,
        needs_onnx_direct_aimet_export: bool = False,
    ):
        self.quant_sim = sim_model
        self.needs_onnx_direct_aimet_export = needs_onnx_direct_aimet_export

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        if target_runtime == TargetRuntime.QNN:
            return SourceModelFormat.ONNX
        else:
            return SourceModelFormat.TORCHSCRIPT

    def quantize(
        self,
        data: _DataLoader,
        num_samples: int | None = None,
        evaluator: BaseEvaluator | None = None,
        device: str = "cpu",
        requantize_model_weights=False,
    ) -> float | None:
        """
        Re-compute quantization encodings for this model with the given dataset and model evaluator.

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
        assert isinstance(self, BaseModel)
        if not evaluator:
            evaluator = self.get_evaluator()

        # Enable or disable quantization for model parameters (model weights).
        # Activations are always re-quantized.
        for quant_module in self.quant_sim.model.modules():
            if isinstance(quant_module, QcQuantizeWrapper):
                for param_quantizer in quant_module.param_quantizers.values():
                    if not requantize_model_weights:
                        try:
                            param_quantizer.freeze_encoding()
                        except RuntimeError:
                            # Encoding is not set, so it can't be frozen.
                            pass
                    else:
                        # Un-freeze the quantizer.
                        param_quantizer._is_encoding_frozen = False

        # Reset evaluator if applicable
        if evaluator:
            evaluator.reset()

        # Define evaluator function for this model.
        def evaluator_func(model: torch.nn.Module, args):
            # This function is defined because AIMET does not unwrap
            # the arguments you pass to `compute_encodings`.
            return (
                evaluator.add_from_dataset(model, *args)
                if evaluator
                else _for_each_batch(model, *args)
            )

        # Compute the new encodings.
        self.quant_sim.compute_encodings(evaluator_func, [data, num_samples, device])

        # Return accuracy score if applicable
        return evaluator.get_accuracy_score() if evaluator else None

    def convert_to_torchscript_and_aimet_encodings(
        self,
        output_dir: str | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
    ) -> str:
        """
        Converts the torch module to a zip file containing an
        unquantized torchscript trace and an aimet quantization encodings file.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        if not input_spec:
            input_spec = self._get_input_spec_ts()

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        base_dir = Path(f"{model_name}.aimet")
        base_path = Path(output_dir) / base_dir
        if base_path.exists():
            shutil.rmtree(base_path)
        os.makedirs(base_path)
        self.quant_sim.export(
            str(base_path),
            model_name,
            tuple(make_torch_inputs(input_spec)),
            export_to_torchscript=True,
        )

        # AIMET exports GraphModule. Convert it to ScriptModule
        fx_graph_path = base_path / f"{model_name}.pth"
        fx_graph = torch.load(fx_graph_path)
        script_module = torch.jit.trace(fx_graph, tuple(make_torch_inputs(input_spec)))
        torch.jit.save(script_module, base_path / f"{model_name}.pt")

        with ZipFile(zip_path, "w") as zip_object:
            zip_object.write(base_path, base_dir)
            zip_object.write(
                base_path / f"{model_name}.pt", base_dir / f"{model_name}.pt"
            )
            zip_object.write(
                base_path / f"{model_name}_torch.encodings",
                base_dir / f"{model_name}_torch.encodings",
            )

        return zip_path

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
    ) -> str:
        """
        Converts the torch module to a zip file containing an
        unquantized ONNX model and an aimet quantization encodings file.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        if not input_spec:
            input_spec = self._get_input_spec_ts()

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        base_dir = Path(f"{model_name}.aimet")
        base_path = Path(output_dir) / base_dir
        if base_path.exists():
            shutil.rmtree(base_path)
        os.makedirs(base_path)

        onnx_utils.EXPORT_TO_ONNX_DIRECT = self.needs_onnx_direct_aimet_export
        self.quant_sim.export(
            str(base_path),
            model_name,
            tuple(make_torch_inputs(input_spec)),
            onnx_export_args=dict(input_names=[name for name in input_spec]),
        )

        onnx_file_name = f"{model_name}.onnx"
        encodings_file_name = f"{model_name}.encodings"
        with ZipFile(zip_path, "w") as zip_object:
            zip_object.write(base_path, base_dir)
            zip_object.write(
                base_path / onnx_file_name, os.path.join(base_dir, onnx_file_name)
            )
            zip_object.write(
                base_path / encodings_file_name,
                os.path.join(base_dir, encodings_file_name),
            )

        return zip_path

    def convert_to_torchscript(*args, **kwargs):
        """Block users from calling convert_to_torchscript() on quantized models, since python will call both parent classes."""
        raise NotImplementedError(
            "This model is quantized. Use `model.convert_to_quantized_torchscript` instead!"
        )

    def convert_to_quantized_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a quantized torchscript trace.
        """
        if not input_spec:
            input_spec = self._get_input_spec_ts()

        with tempfile.TemporaryDirectory() as tempdir:
            self.quant_sim.export(
                tempdir,
                "model",
                tuple(make_torch_inputs(input_spec)),
                export_to_torchscript=True,
                use_embedded_encodings=True,
            )
            return torch.jit.load(f"{tempdir}/model_embedded.torchscript.pth")

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model and input spec.
        Default behavior is randomized input in range [0, 1].
        """
        if not input_spec:
            input_spec = self._get_input_spec_ts()
        inputs = make_torch_inputs(input_spec)
        return {k: v.numpy() for k, v in zip(input_spec.keys(), inputs)}

    def _get_input_spec_ts(self, *args, **kwargs) -> InputSpec:
        """Type safe version of get_input_spec."""
        assert isinstance(self, BaseModel)
        return self.get_input_spec(*args, **kwargs)


class HubCompileOptionsInt8Mixin:
    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
    ) -> str:
        compile_options = super().get_hub_compile_options(  # type: ignore
            target_runtime, other_compile_options
        )
        return compile_options + " --quantize_full_type int8 --quantize_io"
