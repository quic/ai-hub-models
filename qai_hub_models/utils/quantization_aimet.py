# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Items defined in this file require that AIMET be installed.
"""
from __future__ import annotations

import logging
import os

try:
    from aimet_common.utils import AimetLogger  # type: ignore
    from aimet_torch import onnx_utils
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.tensor_quantizer import StaticGridPerTensorQuantizer

    # Suppress aimet info logs within zoo
    if not os.environ.get("SHOW_AIMET_LOGS"):
        AimetLogger.set_level_for_all_areas(logging.WARN)
except (ImportError, ModuleNotFoundError):
    raise NotImplementedError(
        "AIMET must be installed to load quantized models. "
        "AIMET is only supported on Linux. "
        "Install AIMET via the instructions here: "
        "https://quic.github.io/aimet-pages/releases/latest/install/index.html"
    )

import shutil
import tempfile
from pathlib import Path
from typing import Any, List
from zipfile import ZipFile

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.evaluators.base_evaluators import _DataLoader, _for_each_batch
from qai_hub_models.models._shared.common import apply_module_function_recursively
from qai_hub_models.models.common import SourceModelFormat, TargetRuntime
from qai_hub_models.models.protocols import (
    PretrainedHubModelProtocol,
    QuantizableModelProtocol,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs


def tie_aimet_observer_groups(groups: List[List[Any]]):
    """
    This defines groups of ops that all should use the same output
    quantizer observer. The input groups is a list of lists, where the
    inner lists contain op references that should all use the same output
    quantizer. Each op should have an `output_quantizers` member.

    Example:

        groups = [
            [
                sim.model.net.maxpool2,
                sim.model.net.Mixed_5b.module_avg_pool2d,
            ],
        ]
        _tie_aimet_observer_groups(groups)
    """
    for group in groups:
        output_quantizer = group[0].output_quantizers[0]
        for op in group[1:]:
            op.output_quantizers[0] = output_quantizer


def convert_all_depthwise_to_per_tensor(module):
    """
    This recursively iterates a PyTorch module (that has been prepared by
    AIMET for quantization) and replaces the weight quantizers with a
    per-tensor for all depthwise convolutions. All parameters (bitwidth,
    round_mode, etc.) are copied over from the existing quantizer.
    """

    # Please see #9842 for context
    def convert_depthwise_to_per_tensor(op, parent_module, name):
        # Only convert depthwise
        if op.groups > 1 and op.out_channels == op.groups:
            quantizers = parent_module.param_quantizers
            for key in ["weight", "bias"]:
                quantizer = quantizers[key]
                quantizers[key] = StaticGridPerTensorQuantizer(
                    bitwidth=quantizer.bitwidth,
                    round_mode=quantizer.round_mode,
                    quant_scheme=quantizer.quant_scheme,
                    use_symmetric_encodings=quantizer.use_symmetric_encodings,
                    enabled_by_default=quantizer.enabled,
                )

    apply_module_function_recursively(
        module, torch.nn.Conv2d, convert_depthwise_to_per_tensor
    )


class AIMETQuantizableMixin(PretrainedHubModelProtocol, QuantizableModelProtocol):
    """
    Mixin that allows a model to be quantized & exported to disk using AIMET.

    Inheritor must implement HubModel for this mixin to function.
    """

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        needs_onnx_direct_aimet_export: bool = False,
    ):
        self.quant_sim = quant_sim
        self.needs_onnx_direct_aimet_export = needs_onnx_direct_aimet_export

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

        # Define evaluator function for this model.
        def batched_forward(model: torch.nn.Module, args):
            # This function is defined because AIMET does not unwrap
            # the arguments you pass to `compute_encodings`.
            data, num_samples, device = args
            _for_each_batch(model, data, num_samples, device, data_has_gt=data_has_gt)

        # Compute the new encodings.
        self.quant_sim.compute_encodings(batched_forward, [data, num_samples, device])

    def convert_to_torchscript_and_aimet_encodings(
        self,
        output_dir: str | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
    ) -> str:
        if model_name is None:
            model_name = self.__class__.__name__
        if not input_spec:
            input_spec = self.get_input_spec()

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        base_dir = Path(f"{model_name}.aimet")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / base_dir
            os.makedirs(base_path)
            self.quant_sim.export(
                str(base_path),
                model_name,
                tuple(make_torch_inputs(input_spec)),
                export_to_torchscript=True,
            )

            with ZipFile(zip_path, "w") as zip_object:
                zip_object.write(base_path, base_dir)
                zip_object.write(
                    base_path / f"{model_name}.torchscript.pth",
                    base_dir / f"{model_name}.pt",
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
            input_spec = self.get_input_spec()

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        base_dir = Path(f"{model_name}.aimet")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / base_dir
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

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        if not input_spec:
            input_spec = self.get_input_spec()

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
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs = make_torch_inputs(input_spec)
        return {k: v.numpy() for k, v in zip(input_spec.keys(), inputs)}

    def get_hub_compile_options(
        self, target_runtime: TargetRuntime, other_compile_options: str = ""
    ) -> str:
        compile_options = super().get_hub_compile_options(  # type: ignore
            target_runtime, other_compile_options
        )
        return compile_options + " --quantize_full_type int8 --quantize_io"

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX
