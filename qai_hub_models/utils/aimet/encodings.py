# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from copy import deepcopy

import onnx


def propagate_memory_encodings(
    encodings: dict[str, dict],
    model: onnx.ModelProto,
) -> None:
    """
    Propagate encodings through memory ops. This can be important if the
    model will be split up into multiple parts if the split points run
    through ops that do not have encodings and rely on propagation
    downstream. Encodings are updated in place.
    """
    changes = True
    while changes:
        changes = False
        for node in model.graph.node:
            if node.output[0] in encodings["activation_encodings"]:
                continue

            if (
                node.op_type
                in {
                    "Concat",
                    "Split",
                    "Transpose",
                    "Cast",
                    "Reshape",
                    "Slice",
                }
                and node.input[0] in encodings["activation_encodings"]
            ):
                for output_name in node.output:
                    dst_entry = deepcopy(
                        encodings["activation_encodings"][node.input[0]]
                    )
                    if isinstance(dst_entry, dict):
                        dst_entry["name"] = output_name
                    encodings["activation_encodings"][output_name] = dst_entry
                    changes = True
