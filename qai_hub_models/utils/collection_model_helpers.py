# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import ast

from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def get_components(model_id: str) -> list[str] | None:
    """
    Parse the <model_id>/model.py to extract component names from any decorator
    that calls:

    @CollectionModel.add_component(<some_key>, <component>)

    Returns a list of component names (as strings), or None if
    no CollectionModel.add_component call is found.
    """
    model_path = QAIHM_MODELS_ROOT / model_id / "model.py"
    with open(model_path) as f:
        source = f.read()

    tree = ast.parse(source, filename=model_path)
    components: list[str] = []

    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        # We only care about class definitions (which may have decorators)
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    # Check if the decorator calls CollectionModel.add_component
                    if isinstance(decorator.func, ast.Attribute):
                        if (
                            isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "CollectionModel"
                            and decorator.func.attr == "add_component"
                        ):

                            # Expecting exactly one argument: the component.
                            if len(decorator.args) == 1:
                                component_arg = decorator.args[0]
                                if isinstance(component_arg, ast.Name):
                                    components.append(component_arg.id)
                            if len(decorator.args) == 2:
                                component_arg = decorator.args[1]
                                if isinstance(component_arg, ast.Constant):
                                    components.append(component_arg.value)
            if components:
                break  # only check first class defined in the file with added components
    return components if components else None
