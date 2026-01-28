# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator

import onnx
from aimet_onnx.common.defs import (
    QuantizationDataType,
)
from aimet_onnx.qc_quantize_op import (
    GroupedBlockQuantizeDequantize,
    QcQuantizeOp,
)
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx


def _tie_quantizers_for_kv_cache(quantsim_model: QuantSimOnnx) -> None:
    quantizer_mapping = {}

    for input_name in quantsim_model.model.graph().input:
        if "past_key" in input_name.name or "past_value" in input_name.name:
            output_name = input_name.name.replace("in", "out")
            quantizer_mapping[input_name.name] = quantsim_model.qc_quantize_op_dict[
                output_name
            ]
    quantsim_model.set_quantizers(quantizer_mapping)


def _set_tensors_to_output_8b_sym(quantsim_model: QuantSimOnnx) -> None:
    out_tensors = []
    out_tensors.extend(
        [
            t.name
            for t in quantsim_model.model.graph().input
            if "past_key" in t.name or "past_value" in t.name
        ]
    )
    out_tensors.extend(
        [
            t.name.replace("_updated", "")
            for t in quantsim_model.model.graph().output
            if "past_key" in t.name or "past_value" in t.name
        ]
    )
    for out_tensor in out_tensors:
        _set_tensor_to_8_bit_symmetric(quantsim_model, out_tensor)


def _set_4bit_weights_to_lpbq(quantsim_model: QuantSimOnnx) -> None:
    # This is largely a copy-paste of
    # set_grouped_blockwise_quantization_for_weights, except adds an op
    # selection criterion based on all ops that already have the target
    # bitwidth. Can be simplified once that function accepts a function
    # argument.
    block_size = 64
    decompressed_bw = 8
    strict = False
    bitwidth = 4
    for op in quantsim_model.connected_graph.ordered_ops:
        _, _, param_quantizers = quantsim_model.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        if weight_quantizer.bitwidth != bitwidth:
            continue

        try:
            grouped_quantizer = GroupedBlockQuantizeDequantize(
                weight_quantizer.quant_info,
                bitwidth,
                decompressed_bw,
                block_size,
                weight_quantizer.quant_scheme,
                weight_quantizer.op_mode,
                weight_quantizer.tensor_quantizer_params,
            )
        except ValueError:
            if strict:
                raise
        else:
            if bias_quantizer:
                bias_quantizer.enable_per_channel_quantization()
                bias_quantizer.use_symmetric_encodings = True
                bias_quantizer.data_type = QuantizationDataType.int

            for name, quantizer in quantsim_model.qc_quantize_op_dict.items():
                if quantizer is weight_quantizer:
                    quantsim_model.qc_quantize_op_dict[name] = grouped_quantizer


def _set_tensor_to_8_bit_symmetric(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> None:
    if tensor_name in quantsim_model.qc_quantize_op_dict:
        quantizer = quantsim_model.qc_quantize_op_dict[tensor_name]
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True


def _set_lm_head_to_8b(quantsim_model: QuantSimOnnx) -> None:
    for weight in _get_lm_head_weights(quantsim_model.model.model):
        quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
        quantizer.set_bitwidth(8)
        quantizer.quant_info.blockSize = 0
        quantizer.quant_info.blockAxis = -1
        quantizer.enable_per_channel_quantization()


def _get_lm_head_weights(
    model: onnx.ModelProto,
) -> Generator[onnx.TensorProto, None, None]:
    vocab_size = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    for weight in model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            for node in model.graph.node:
                if node.op_type in ("Gemm", "MatMul", "Conv") and node.input[1] in {
                    weight.name,
                    weight.name + "_updated",
                    weight.name + "_qdq",
                }:
                    yield weight
