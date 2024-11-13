# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Implementation of a class that splits a larger onnx graph into smaller subgraphs

import collections
import os

import onnx
from onnx.external_data_helper import uses_external_data


class OnnxSplitter:
    def __init__(self, onnxmodel, verbose=False):
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
        name,  # name of the ONNX graph
        output_tensors,  # list of new output tensors to include
        additional_input_tensors=None,
    ):
        """
        Partition a graph with input and output tensors
        - Captures all nodes that required to compute the given output_tensors
        """

        def upstream(nodeid):
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

        use = set()
        for nodeid in visited:
            use.update(self.node[nodeid].input)
            use.update(self.node[nodeid].output)

        # Include in-use items and preserve the original order
        new_node = [i for i in self.model.graph.node if id(i) in visited]
        new_initializer = [i for i in self.model.graph.initializer if i.name in use]
        new_value_info = [i for i in self.model.graph.value_info if i.name in use]
        new_sparse_initializer = [
            i for i in self.model.graph.sparse_initializer if i.name in use
        ]

        value_info_dict = {i.name: i for i in new_value_info}
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

    def split(self, list_of_intermediate_output_tensors):
        count = 0
        additional_input_tensors, covered_output_tensors = [], set()
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
    def get_all_tensors(cls, graph):
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
    def is_using_external_data(cls, onnxmodel):
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
        if os.path.exists(os.path.join(dirname, kwargs["location"])):
            os.unlink(os.path.join(dirname, kwargs["location"]))

    onnx.save(model, newonnxfile, **kwargs)


def split_onnx_by_names(
    onnxfile, list_of_output_tensors, output_dir=".", verbose=False
):
    if verbose:
        print(f"Loading {onnxfile}")
    onnxmodel = onnx.load(onnxfile, load_external_data=False)
    splitter = OnnxSplitter(onnxmodel, verbose=verbose)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    list_of_output_tensors = [i.split(",") for i in list_of_output_tensors]
    num_splits = len(list_of_output_tensors) + 1

    # 1. split model
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f"{os.path.basename(onnxfile)}_{i + 1}_of_{num_splits}"
        input_tensors = [i.name for i in subgraph.input]
        new_model_info.append([new_basename, input_tensors])

        submodel = onnx.helper.make_model(
            subgraph, opset_imports=onnxmodel.opset_import
        )
        if (
            not using_external_data
            and submodel.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF
        ):
            onnx.checker.check_model(submodel)

        if using_external_data:
            if verbose:
                print(f"Loading external data from {os.path.dirname(onnxfile)}")
            onnx.load_external_data_for_model(
                submodel, base_dir=os.path.dirname(onnxfile)
            )

        newonnxfile = f"{output_dir}/{new_basename}.onnx"
        if verbose:
            print(f"Saving {newonnxfile}")
        save_model(submodel, newonnxfile, using_external_data)
