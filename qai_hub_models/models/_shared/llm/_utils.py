# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx


def _tie_quantizers_for_kv_cache(quantsim_model: QuantSimOnnx) -> None:
    quantizer_mapping = dict()
    for input_name in quantsim_model.model.graph().input:
        if "past_key" in input_name.name or "past_value" in input_name.name:
            output_name = input_name.name.replace("in", "out")
            quantizer_mapping[input_name.name] = quantsim_model.qc_quantize_op_dict[
                output_name
            ]
    quantsim_model.set_quantizers(quantizer_mapping)


def _set_tensors_to_output_8b_sym(quantsim_model: QuantSimOnnx):
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


def _set_tensor_to_8_bit_symmetric(quantsim_model: QuantSimOnnx, tensor_name: str):
    if tensor_name in quantsim_model.qc_quantize_op_dict:
        quantizer = quantsim_model.qc_quantize_op_dict[tensor_name]
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True


def _set_lm_head_to_8b(quantsim_model: QuantSimOnnx):
    for weight in _get_lm_head_weights(quantsim_model.model.model):
        quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
        quantizer.set_bitwidth(8)
        quantizer.quant_info.blockSize = 0
        quantizer.quant_info.blockAxis = -1
        quantizer.enable_per_channel_quantization()


def _get_lm_head_weights(quantsim_model: QuantSimOnnx):
    vocab_size = quantsim_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    for weight in quantsim_model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            for node in quantsim_model.graph.node:
                if node.op_type in ("Gemm", "MatMul", "Conv") and node.input[1] in {
                    weight.name,
                    weight.name + "_updated",
                    weight.name + "_qdq",
                }:
                    yield weight
