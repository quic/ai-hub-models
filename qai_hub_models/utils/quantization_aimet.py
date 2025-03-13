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
import sys
from contextlib import contextmanager

try:
    import aimet_torch.nn.modules.custom as aimet_ops
    from aimet_common import quantsim
    from aimet_common.connected_graph.operation import Op as AimetOp
    from aimet_common.connected_graph.product import Product as AimetProduct
    from aimet_common.utils import AimetLogger
    from aimet_torch import onnx_utils
    from aimet_torch.v1.qc_quantize_op import QcQuantizeOpMode, QcQuantizeWrapper
    from aimet_torch.v1.quantsim import QuantizationSimModel
    from aimet_torch.v1.tensor_quantizer import StaticGridPerTensorQuantizer

    # TODO: Temporarily falling back to 0.6.1 encoding format since
    # the new default encoding schema (1.0.0) is not supported in AI Hub yet.
    quantsim.encoding_version = "0.6.1"

    # Suppress aimet info logs within zoo
    if not os.environ.get("SHOW_AIMET_LOGS"):
        AimetLogger.set_level_for_all_areas(logging.WARN)
except (ImportError, ModuleNotFoundError) as e:
    if "libcublas.so.11" in e.msg:
        raise NotImplementedError(
            "Missing library 'libcublas.so.11'. Run `sudo apt install libcublas11` to resolve."
        )
    raise NotImplementedError(
        "Quantized models require the AIMET package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional
from zipfile import ZipFile

import torch
import torch.nn.modules as nn
from onnx import load_model as load_onnx_model
from onnx import save_model as save_onnx_model
from qai_hub.client import DatasetEntries, Device

from qai_hub_models.evaluators.base_evaluators import _DataLoader, _for_each_batch
from qai_hub_models.models._shared.common import apply_module_function_recursively
from qai_hub_models.models.common import Precision, SourceModelFormat, TargetRuntime
from qai_hub_models.models.protocols import (
    PretrainedHubModelProtocol,
    QuantizableModelProtocol,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs


@contextmanager
def redirect_stderr():
    """
    Can't use built-in functions like contextlib.redirect_stderr
    because the stderr is coming from cpp extensions.
    """
    with open(os.devnull, "w") as f:
        original_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_fd)
        try:
            os.dup2(f.fileno(), original_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, original_fd)
            os.close(saved_stderr_fd)


def _should_tie_observers(op: torch.nn.Module) -> bool:
    """
    Determine whether the input and output observers of this op should be tied.
    """
    if not hasattr(op, "_module_to_wrap"):
        return False
    wrapped_op = op._module_to_wrap
    op_types_to_tie = [
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.Upsample,
        aimet_ops.Concat,
        aimet_ops.Interpolate,
        aimet_ops.MaxPool2d,
    ]
    for op_type in op_types_to_tie:
        if isinstance(wrapped_op, op_type):
            return True
    return False


def _get_observer_module_name(modules: dict[str, Any], target: Any) -> Optional[str]:
    if not isinstance(target, str):
        return None
    module = modules.get(target)
    if isinstance(module, QcQuantizeWrapper):
        return target
    elif isinstance(module, aimet_ops.CustomSiLU):
        return target + ".mul"
    return None


def _tie_quantizer_deps(
    quantizer_deps: dict[str, list[str]], modules: dict[str, torch.nn.Module]
) -> None:
    """
    Given a dependency graph of nodes, tie output quantizers of nodes that share an edge.
    All edges should be bidirectional.
    """
    seen = set()
    for input_module_name in quantizer_deps.keys():
        if input_module_name in seen:
            continue
        seen.add(input_module_name)
        stack = [input_module_name]
        module = modules[input_module_name]
        assert isinstance(module, QcQuantizeWrapper)
        quantizer = module.output_quantizers[0]
        while len(stack) > 0:
            curr_node = stack.pop()
            for edge in quantizer_deps[curr_node]:
                if edge in seen:
                    continue
                seen.add(edge)
                stack.append(edge)
                edge_module = modules[edge]
                assert isinstance(edge_module, QcQuantizeWrapper)
                edge_module.output_quantizers[0] = quantizer


def tie_observers(quant_sim: QuantizationSimModel) -> None:
    """
    For certain ops, the input and output observers need to be the same.

    For example, in a concat op, all the inputs need to have the same scale.
    Otherwise, there will need to be an additional explicit quantize which is lossy.

    Other ops like MaxPool and AvgPool are constraints in tflite.

    This assumes all modules have exactly one output.
    All modules (i.e. instances of `StaticGridQuantWrapper`) have both an
        `output_quantizers` and `input_quantizers` field.
    We only update output_quantizers since empircally that's all that seems to matter.
    """
    fx_graph = quant_sim.model
    nodes = [node for node in fx_graph.graph.nodes]
    modules = dict(fx_graph.named_modules())

    # quant_sim.model is a torch fx graph. This graph stores nodes and modules.
    # nodes store the graph structure (which ops input into other ops).
    # modules store the op objects themselves.
    # Create a dependency graph of modules that need to share output quantizers.
    quantizer_deps: dict[str, list[str]] = {}
    for node in nodes:
        module = modules.get(node.target)
        if module is None or not _should_tie_observers(module):
            continue
        if node.target not in quantizer_deps:
            quantizer_deps[node.target] = []
        for input_node in node.all_input_nodes:
            # If the node is something like a reshape with no observers,
            # Keep going up the tree until you find something with an observer.
            # If one of these nodes has multiple inputs, this may behave incorrectly.
            while (
                observer_module_name := _get_observer_module_name(
                    modules, input_node.target
                )
            ) is None:
                if input_node.target == getattr:
                    # If the input node is getting a tensor attribute (e.g. shape)
                    # No observers need to be tied
                    break
                input_node = input_node.all_input_nodes[0]
            if input_node.target == getattr or observer_module_name is None:
                continue
            if observer_module_name not in quantizer_deps:
                quantizer_deps[observer_module_name] = []
            quantizer_deps[observer_module_name].append(node.target)
            quantizer_deps[node.target].append(observer_module_name)
    _tie_quantizer_deps(quantizer_deps, modules)


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


def constrain_quantized_inputs_to_range(
    qsim: QuantizationSimModel, range: tuple[float, float]
):
    """
    For all model inputs, set the quantizer to have the provided input range.
    """

    # Map: <nn.Module, List of Quantized Inputs>
    module_to_inputs_idx: dict[torch.nn.Module, list[int]] = {}
    op: AimetOp
    for op in qsim.connected_graph.get_all_ops().values():
        if op.get_module():
            module = op.get_module()
            idx: int
            input_product: AimetProduct
            for idx, input_product in enumerate(op.get_input_products()):
                if input_product.is_model_input:
                    if module in module_to_inputs_idx:
                        module_to_inputs_idx[module].append(idx)
                    else:
                        module_to_inputs_idx[module] = [idx]

    for _, quant_wrapper in qsim.quant_wrappers():
        if indices := module_to_inputs_idx.get(quant_wrapper._module_to_wrap):
            for index in indices:
                if index < len(quant_wrapper.input_quantizers):
                    quant_wrapper.input_quantizers[
                        index
                    ].encoding_min_max_fixed_vals = range


def constrain_quantized_inputs_to_image_range(qsim: QuantizationSimModel):
    """
    For all model inputs, set the quantizer to have a range of (0, 1).

    The range translates to the full image range for int8/uint8/int16/uint16.

    This is typically useful for use in pipelines where the input is already int8/uint8 and
    we know the quantization range beforehand (eg. a RGB camera feed).
    """
    return constrain_quantized_inputs_to_range(qsim, (0.0, 1.0))


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

    def load_encodings(
        self,
        encodings: Mapping | str | os.PathLike,
        strict: bool = True,
        partial: bool = True,
        requires_grad: bool | None = None,
        allow_overwrite: bool | None = True,
    ):
        for _, module in self.quant_sim.quant_wrappers():
            module.set_mode(QcQuantizeOpMode.ACTIVE)

        self.quant_sim.load_encodings(
            encodings=encodings,
            strict=strict,
            partial=partial,
            requires_grad=requires_grad,
            allow_overwrite=allow_overwrite,
        )

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

        with qaihm_temp_dir() as tmpdir:
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
        external_weights: bool = False,
        output_names: Optional[list[str]] = None,
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

        with qaihm_temp_dir() as tmpdir:
            base_path = Path(tmpdir) / base_dir
            if base_path.exists():
                shutil.rmtree(base_path)
            os.makedirs(base_path)

            onnx_utils.EXPORT_TO_ONNX_DIRECT = self.needs_onnx_direct_aimet_export

            with redirect_stderr():
                self.quant_sim.export(
                    str(base_path),
                    model_name,
                    tuple(make_torch_inputs(input_spec)),
                    onnx_export_args=dict(
                        input_names=[name for name in input_spec],
                        output_names=output_names,
                    ),
                )

            onnx_file_path = str(base_path / f"{model_name}.onnx")
            encoding_file_path = str(base_path / f"{model_name}.encodings")
            external_weights_file_path = ""

            if external_weights:
                # Torch exports to onnx with external weights scattered in a directory.
                # Save ONNX model with weights to one file.
                external_weights_file_name = f"{model_name}.data"
                external_weights_file_path = str(base_path / external_weights_file_name)
                onnx_model = load_onnx_model(onnx_file_path)
                save_onnx_model(
                    onnx_model,
                    str(onnx_file_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=external_weights_file_name,
                )

            zip_aimet_model(
                zip_path,
                base_dir,
                onnx_file_path,
                encoding_file_path,
                external_weights_file_path,
            )

        return zip_path

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        if not input_spec:
            input_spec = self.get_input_spec()

        with qaihm_temp_dir() as tempdir:
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
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model and input spec.
        """
        return None

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = ""
        if precision.activations_type is not None:
            quantization_flags = " --quantize_io"
            if target_runtime == TargetRuntime.TFLITE:
                # uint8 is the easiest I/O type for integration purposes,
                # especially for image applications. Images are always
                # uint8 RGB when coming from disk or a camera.
                #
                # Uint8 has not been thoroughly tested with other paths,
                # so it is enabled only for TF Lite today.
                quantization_flags += " --quantize_io_type uint8"

        return (
            super().get_hub_compile_options(  # type: ignore[safe-super]
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        return SourceModelFormat.ONNX

    def __call__(self, *args, **kwargs):
        """
        Instance of AIMETQuantizableMixin should never be trained,
            so should be safe to disable gradients during forward pass.
        """
        with torch.no_grad():
            return super().__call__(*args, **kwargs)  # type: ignore[misc]

    def get_hub_quantize_options(self, precision: Precision) -> str:
        """
        AI Hub quantize options recommended for the model.
        """
        return ""
