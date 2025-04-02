# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import re
from copy import deepcopy

import onnx


def find_name_mapping(pattern_pairs, src_names, dst_names, dst_input_names=None):
    patterns = [re.compile(x) for x, y in pattern_pairs]
    mapping = {}
    rev_mapping = {}
    known_unused = set()
    for src_name in src_names:
        for i in range(len(pattern_pairs)):
            m = patterns[i].match(src_name)
            if m:
                dst_patterns = pattern_pairs[i][1]
                if not isinstance(dst_patterns, list):
                    dst_patterns = [dst_patterns]

                used = False
                for dst_pattern in dst_patterns:
                    if isinstance(dst_pattern, tuple):
                        assert dst_input_names is not None
                        # This contains a (node, index) pair, where the index
                        # refers to the input index of that node
                        dst_pattern, index = dst_pattern
                        dst_name = dst_pattern.format(*m.groups())
                        if dst_name in dst_input_names:
                            real_dst_name = dst_input_names[dst_name][index]
                            mapping[src_name] = real_dst_name
                            rev_mapping[real_dst_name] = src_name
                            used = True

                    elif not dst_pattern:
                        known_unused.add(src_name)
                        used = True
                    else:
                        # This dst_name refers to the edge name
                        dst_name = dst_pattern.format(*m.groups())
                        if dst_name in dst_names:
                            mapping[src_name] = dst_name
                            rev_mapping[dst_name] = src_name
                            used = True
                if used:
                    break

    return mapping, rev_mapping, known_unused


def map_encodings(
    pattern_pairs,
    src_names,
    dst_names,
    dst_input_names=None,
    src_encodings=[],
    dst_encodings=[],
):
    patterns = [re.compile(x) for x, y in pattern_pairs]
    mapping = {}
    rev_mapping = {}
    known_unused = set()

    def default_callback(
        src_encodings,
        dst_encodings,
        src_name,
        dst_name,
        pattern_index,
        num_patterns,
        groups,
    ):
        if src_name in src_encodings:
            src_entry = src_encodings[src_name]
            dst_entry = deepcopy(src_entry)
            if isinstance(dst_entry, dict):
                dst_entry["name"] = dst_name
            dst_encodings[dst_name] = dst_entry

    for src_name in src_names:
        for i in range(len(pattern_pairs)):
            m = patterns[i].match(src_name)
            if m:
                dst_patterns = pattern_pairs[i][1]
                callback = default_callback

                if isinstance(dst_patterns, tuple) and callable(dst_patterns[1]):
                    dst_patterns, callback = dst_patterns

                if not isinstance(dst_patterns, list):
                    dst_patterns = [dst_patterns]

                used = False

                for dst_pattern_index, dst_pattern in enumerate(dst_patterns):
                    if isinstance(dst_pattern, tuple):
                        assert dst_input_names is not None
                        # This contains a (node, index) pair, where the index
                        # refers to the input index of that node
                        dst_pattern, index = dst_pattern
                        dst_name = dst_pattern.format(*m.groups())
                        if dst_name in dst_input_names:
                            real_dst_name = dst_input_names[dst_name][index]
                            mapping[src_name] = real_dst_name
                            rev_mapping[real_dst_name] = src_name
                            used = True

                            callback(
                                src_encodings,
                                dst_encodings,
                                src_name,
                                real_dst_name,
                                dst_pattern_index,
                                len(dst_patterns),
                                m.groups(),
                            )

                    elif not dst_pattern:
                        known_unused.add(src_name)
                        used = True
                    else:
                        # This dst_name refers to the edge name
                        dst_name = dst_pattern.format(*m.groups())
                        if dst_name in dst_names:
                            mapping[src_name] = dst_name
                            rev_mapping[dst_name] = src_name
                            used = True

                            callback(
                                src_encodings,
                                dst_encodings,
                                src_name,
                                dst_name,
                                dst_pattern_index,
                                len(dst_patterns),
                                m.groups(),
                            )
                if used:
                    break

    return mapping, rev_mapping, known_unused


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

            if node.op_type in {
                "Concat",
                "Split",
                "Transpose",
                "Cast",
                "Reshape",
                "Slice",
            }:
                if node.input[0] in encodings["activation_encodings"]:
                    for output_name in node.output:
                        dst_entry = deepcopy(
                            encodings["activation_encodings"][node.input[0]]
                        )
                        if isinstance(dst_entry, dict):
                            dst_entry["name"] = output_name
                        encodings["activation_encodings"][output_name] = dst_entry
                        changes = True
