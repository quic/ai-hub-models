# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from aimet_onnx.meta.connectedgraph import Op
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import QuantizationSimModel


def _candidate_names(op: Op) -> tuple[str, str]:
    """
    Candidate strings to match exception patterns against for a connected-graph Op.

    ONNX node names are path-like (e.g. "/decoder/self_attn/k_proj"), while the
    patterns in exceptions.json were written for underscore-flattened name+optype
    strings (e.g. "k_proj_MatMul"). Op fusion (Linear+Add -> Gemm) can also remove
    the optype suffix the pattern expects, so try both the flattened name and the
    flattened name with the op type appended.
    """
    flat = op.dotted_name.lstrip("/").replace("/", "_")
    return (flat, f"{flat}_{op.type}")


def apply_input_output_exception(
    quantizer: QcQuantizeOp | None, exception: dict[str, Any]
) -> QcQuantizeOp | None:
    assert isinstance(exception, dict), (
        f"Exception is not a dictionary type: {exception}"
    )

    if quantizer is None:
        return None

    assert "encoding_overrides" not in exception, (
        "encoding_overrides is not supported for aimet_onnx quantizers"
    )

    if exception.get("enabled", True) is False:
        quantizer.enabled = False
        return quantizer

    if "bitwidth" in exception:
        quantizer.set_bitwidth(exception["bitwidth"])

    if "asymmetric" in exception:
        quantizer.use_symmetric_encodings = exception["asymmetric"] is False

    return quantizer


def apply_param_exception(
    quantizer: QcQuantizeOp, exception: dict[str, Any]
) -> QcQuantizeOp:
    assert quantizer
    assert isinstance(exception, dict), (
        f"Exception is not a dictionary type: {exception}"
    )

    if exception.get("enabled", True) is False:
        quantizer.enabled = False
        return quantizer

    quantizer.use_symmetric_encodings = exception.get("asymmetric", False) is False
    quantizer.set_bitwidth(exception.get("bitwidth", 4))

    return quantizer


def get_weight_param_name(op: Op) -> str | None:
    """Returns the name of the tensor holding op's "weight" parameter, or None."""
    for name, (_, param_type) in op.parameters.items():
        if param_type == "weight":
            return name

    # aimet_onnx's connected graph doesn't tag Gather's embedding table as a
    # "parameter" product, but exceptions.json's *_Gather rules target it.
    if op.type == "Gather" and op.inputs:
        return op.inputs[0].name

    return None


class ManualQuantsimMixedPrecisionConfig:
    def __init__(self, mixed_precision_config_file: str) -> None:
        exception_types = ("name", "module")
        exceptions: dict[str, dict] = {k: {} for k in exception_types}

        with open(mixed_precision_config_file) as f:
            exception_config = json.load(f)

            # Populate op_list here
            for etype in exception_types:
                for item in exception_config[f"{etype}_list"]:
                    expections_str = {
                        k: v for k, v in item["exceptions"].items() if v is not None
                    }
                    print(f"Applying {item['module_name']}:\t{expections_str}")
                    exceptions[etype][item["module_name"]] = item["exceptions"]

        self.exceptions_dict = exceptions

    def apply_exceptions(self, quant_sim: QuantizationSimModel) -> None:
        matched_counts: defaultdict[str, int] = defaultdict(int)

        for op in quant_sim.connected_graph.ordered_ops:
            candidates = _candidate_names(op)

            for etype in ("module", "name"):
                exception = None
                matched_key: str = ""

                for key, values in self.exceptions_dict[etype].items():
                    if etype == "module":
                        if key == op.type:
                            exception, matched_key = values, key
                            break
                    else:
                        for candidate in candidates:
                            if re.search(key, candidate):
                                exception, matched_key = values, key
                                break

                if exception is None:
                    continue

                matched_counts[matched_key] += 1

                if exception["param_exceptions"] is not None:
                    self.apply_param_exception_to_op(
                        exception["param_exceptions"], op, quant_sim
                    )

                if exception["input_exceptions"] is not None:
                    self.apply_input_exception_to_op(
                        exception["input_exceptions"], op, quant_sim
                    )

                if exception["output_exceptions"] is not None:
                    self.apply_output_exception_to_op(
                        exception["output_exceptions"], op, quant_sim
                    )

        for etype in ("module", "name"):
            for key in self.exceptions_dict[etype]:
                if matched_counts[key] == 0:
                    print(
                        f"WARNING: exception pattern {key!r} matched no ops in the quantsim"
                    )

    def apply_param_exception_to_op(
        self,
        param_exceptions: dict[str, Any],
        op: Op,
        quant_sim: QuantizationSimModel,
    ) -> None:
        weight_name = get_weight_param_name(op)
        if weight_name is None:
            return

        quantizer = quant_sim.qc_quantize_op_dict.get(weight_name)
        if quantizer is not None:
            apply_param_exception(quantizer, param_exceptions)

    def apply_input_exception_to_op(
        self,
        input_exceptions: list[dict[str, Any]],
        op: Op,
        quant_sim: QuantizationSimModel,
    ) -> None:
        for exception in input_exceptions:
            index = exception["input_index"]
            if index >= len(op.inputs):
                continue

            quantizer = quant_sim.qc_quantize_op_dict.get(op.inputs[index].name)
            apply_input_output_exception(quantizer, exception)

    def apply_output_exception_to_op(
        self,
        output_exceptions: list[dict[str, Any]],
        op: Op,
        quant_sim: QuantizationSimModel,
    ) -> None:
        for exception in output_exceptions:
            index = exception["output_index"]
            if index >= len(op.outputs):
                continue

            quantizer = quant_sim.qc_quantize_op_dict.get(op.outputs[index].name)
            apply_input_output_exception(quantizer, exception)
