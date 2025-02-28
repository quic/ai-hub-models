# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Implementation of a class that splits a larger onnx graph into smaller subgraphs
from __future__ import annotations

import collections
import os
from collections.abc import Iterable, Iterator

import onnx
from onnx.external_data_helper import uses_external_data


class OnnxSplitter:
    def __init__(self, onnxmodel: onnx.ModelProto, verbose=False):
        self.model = onnxmodel
        self.verbose = verbose
        self.graph_inputs = {i.name for i in self.model.graph.input}
        self.graph_outputs = {i.name for i in self.model.graph.output}
        # nodeid:Onnx Node
        self.node = {id(node): node for node in self.model.graph.node}
        # tensorname: nodeid
        self.producer = {
            output: id(node) for node in self.model.graph.node for output in node.output
        }

    def partition_subgraph(
        self,
        name: str,  # name of the ONNX graph
        output_tensors: Iterable[str],  # list of new output tensors to include
        additional_input_tensors: Iterable[str] | None = None,
    ):
        """
        Partition a graph with input and output tensors
        - Captures all nodes that required to compute the given output_tensors
        """

        def upstream(nodeid: int) -> list[int]:
            return [
                self.producer[i]
                for i in self.node[nodeid].input
                if i not in leaf_tensors
            ]

        # Check prerequisite
        value_info = {i.name: i for i in self.model.graph.value_info}
        assert all(
            [
                (name in value_info) or (name in self.graph_outputs)
                for name in output_tensors
            ]
        ), "ValueInfoProto of output_tensors should be given"

        # prepare the 'leaf' tensors, which can be model input or parameter tensors
        leaf_tensors = set(self.graph_inputs)
        leaf_tensors.update({i.name for i in self.model.graph.initializer})
        if additional_input_tensors is not None:
            leaf_tensors.update(additional_input_tensors)
            self.graph_inputs.update(additional_input_tensors)

        visited_output_tensors, visited_input_tensors = set(output_tensors), set()

        # Traverse from output_tensors to input or 'leaf' nodes
        q = collections.deque([self.producer[i] for i in output_tensors])
        visited = set()
        while q:
            nodeid = q.popleft()
            if nodeid in visited:
                continue
            visited.add(nodeid)
            visited_output_tensors.update(
                [i for i in self.node[nodeid].output if i in self.graph_outputs]
            )
            visited_input_tensors.update(
                [i for i in self.node[nodeid].input if i in self.graph_inputs]
            )
            for producerid in upstream(nodeid):
                if producerid not in visited:
                    q.append(producerid)

        use: set[str] = set()
        for nodeid in visited:
            use.update(self.node[nodeid].input)
            use.update(self.node[nodeid].output)

        # Include in-use items and preserve the original order
        new_node = [i for i in self.model.graph.node if id(i) in visited]
        new_initializer = [i for i in self.model.graph.initializer if i.name in use]
        new_value_info_incl_io = [
            i for i in self.model.graph.value_info if i.name in use
        ]
        new_sparse_initializer = [
            i for i in self.model.graph.sparse_initializer if i.name in use
        ]

        value_info_dict = {i.name: i for i in new_value_info_incl_io}
        value_info_dict.update({i.name: i for i in self.model.graph.output})
        if additional_input_tensors is not None:
            new_inputs = [
                value_info_dict[i]
                for i in additional_input_tensors
                if i in value_info_dict and i in use
            ]
        else:
            new_inputs = []
        new_inputs += [i for i in self.model.graph.input if i.name in use]

        new_outputs = [value_info_dict[i] for i in output_tensors]
        new_outputs += [
            value_info_dict[i.name]
            for i in self.model.graph.output
            if i.name in visited_output_tensors and i.name not in output_tensors
        ]

        io_names = {i.name for i in new_inputs + new_outputs}

        # do not include IO in value_info (this is not proper in ONNX)
        new_value_info = [i for i in new_value_info_incl_io if i.name not in io_names]

        if self.verbose:
            print("new_inputs", [i.name for i in new_inputs])
        if self.verbose:
            print("new_outputs", [i.name for i in new_outputs])
        new_graph = onnx.helper.make_graph(
            nodes=new_node,
            name=name,
            inputs=new_inputs,
            outputs=new_outputs,
            initializer=new_initializer,
            value_info=new_value_info,
            sparse_initializer=new_sparse_initializer,
        )
        return new_graph

    def split(
        self, list_of_intermediate_output_tensors: Iterable[str]
    ) -> Iterator[onnx.GraphProto]:
        count = 0
        additional_input_tensors: list[str] = []
        covered_output_tensors: set[str] = set()
        for i, output_tensors in enumerate(list_of_intermediate_output_tensors):
            count += 1
            graphname = f"{self.model.graph.name}_split{count}"
            if self.verbose:
                print(f"Partitoin new graph: {graphname} for outputs[{output_tensors}]")
            subgraph = self.partition_subgraph(
                graphname, output_tensors, additional_input_tensors
            )
            additional_input_tensors += [
                i for i in output_tensors if i not in self.graph_outputs
            ]
            covered_output_tensors.update([i.name for i in subgraph.output])
            yield subgraph

        graphname = f"{self.model.graph.name}_split{count + 1}"
        last_output_tensors = [
            i.name
            for i in self.model.graph.output
            if i.name not in covered_output_tensors
        ]
        lastgraph = self.partition_subgraph(
            graphname, last_output_tensors, additional_input_tensors
        )
        yield lastgraph

    @classmethod
    def get_all_tensors(cls, graph: onnx.GraphProto):
        yield from graph.initializer
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from cls.get_all_tensors(attribute.g)
                if attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        yield from cls.get_all_tensors(graph)
                if attribute.HasField("t"):
                    yield attribute.t
                yield from attribute.tensors

    @classmethod
    def is_using_external_data(cls, onnxmodel: onnx.ModelProto):
        for tensor in cls.get_all_tensors(onnxmodel.graph):
            if uses_external_data(tensor):
                return True
        return False


def save_model(model, newonnxfile, using_external_data=False):
    kwargs = {}
    if using_external_data or model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        dirname = os.path.dirname(newonnxfile)
        location = os.path.basename(newonnxfile).replace(".onnx", ".data")
        kwargs["save_as_external_data"] = True
        kwargs["all_tensors_to_one_file"] = True
        kwargs["location"] = location
        if os.path.exists(os.path.join(dirname, str(kwargs["location"]))):
            os.unlink(os.path.join(dirname, str(kwargs["location"])))

    # Older ONNX versions treat `location` parameter relative to cwd and not
    # the model file. To avoid inconsistent behavior across ONNX versions, we
    # align cwd and the model directory.
    old_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(newonnxfile))
        onnx.save(model, os.path.basename(newonnxfile), **kwargs)  # type: ignore[arg-type]
    finally:
        os.chdir(old_cwd)
