# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Upload the model's source to AI Hub."""

from __future__ import annotations

import tempfile

import qai_hub as hub

from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)
from qai_hub_models.utils.input_spec import InputSpec


def upload_source(model: BaseModel, input_spec: InputSpec | None = None) -> hub.Model:
    """Serialize the model and upload it to AI Hub."""
    input_spec = input_spec or model.get_input_spec()
    with tempfile.TemporaryDirectory() as tmpdir:
        return hub.upload_model(str(model.serialize(tmpdir, input_spec)))


def upload_collection_source(
    model: CollectionModel,
    input_specs: dict[str, InputSpec] | None = None,
    components: list[str] | None = None,
) -> ComponentGroup[hub.Model]:
    """Serialize and upload each component's source model."""
    input_specs = input_specs or model.get_input_spec()
    components = components or model.component_names
    uploaded: ComponentGroup[hub.Model] = ComponentGroup()
    for name in components:
        spec = input_specs[name]
        with tempfile.TemporaryDirectory() as tmpdir:
            uploaded[name] = hub.upload_model(
                str(model.serialize_component(name, tmpdir, spec))
            )
    return uploaded


def upload_multi_graph_source(
    model: MultiGraphWorkbenchModel,
    input_specs: MultiGraphGroup[InputSpec] | None = None,
) -> MultiGraphGroup[hub.Model]:
    """
    Serialize and upload each graph's source.

    When ``model.shared_source_model`` is True the first uploaded source is
    reused across all remaining graphs.
    """
    input_specs = input_specs or model.get_input_spec()
    uploaded: MultiGraphGroup[hub.Model] = MultiGraphGroup()
    shared: hub.Model | None = None
    for graph_name, spec in input_specs.items():
        if shared is not None:
            uploaded[graph_name] = shared
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_model = hub.upload_model(
                str(model.serialize_graph(graph_name, tmpdir, spec))
            )
        uploaded[graph_name] = hub_model
        if model.shared_source_model:
            shared = hub_model
    return uploaded


def upload_multi_graph_collection_source(
    model: MultiGraphCollectionModel,
    input_specs: MultiGraphComponentGroup[InputSpec] | None = None,
    components: list[str] | None = None,
) -> MultiGraphComponentGroup[hub.Model]:
    """
    Serialize and upload each ``(component, graph)`` pair.

    Components whose graphs share one source file are uploaded once and the
    resulting hub model is reused across all graphs of that component.
    """
    input_specs = input_specs or model.get_input_spec()
    components = components or model.component_names
    uploaded: MultiGraphComponentGroup[hub.Model] = MultiGraphComponentGroup()
    shared: dict[str, hub.Model] = {}
    for (comp_name, graph_name), spec in input_specs.items():
        if comp_name not in components:
            continue
        if comp_name in shared:
            uploaded[(comp_name, graph_name)] = shared[comp_name]
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_model = hub.upload_model(
                str(
                    model.serialize_component_graph(comp_name, graph_name, tmpdir, spec)
                )
            )
        uploaded[(comp_name, graph_name)] = hub_model
        if model.get_component_has_shared_source_model(comp_name):
            shared[comp_name] = hub_model
    return uploaded
