# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.models._shared.llm.model import LLMInstantiationType
from qai_hub_models.utils.onnx_helpers import ONNXBundle

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx


@dataclass
class LLMInstantiation:
    """
    Metadata for an LLM "Instantiation"

    Export instantiates 2 copies of the LLM, each with a different sequence length.
    Both are exported to ONNX files. This is the metadata for each instantiation.
    """

    type: LLMInstantiationType
    model: LLM_AIMETOnnx

    # Exported, unsplit ONNX Bundle for this model.
    onnx_bundle: ONNXBundle | None = None

    # Outputs.
    device_output: DatasetEntries | None = None
    gt_output: list[np.ndarray] | None = None

    @property
    def name(self) -> str:
        return self.type.value


@dataclass
class LLMComponent:
    """
    An LLM is broken into N "components", which are often called "parts" or "splits".
    Each component will become a separate context binary in the final model bundle.

    This is the metadata for a single component.
    """

    component_idx: int
    link_job: hub.client.LinkJob | None = None

    subcomponent_onnx_model: dict[LLMInstantiationType, ONNXBundle] = field(
        default_factory=dict
    )
    subcomponent_compile_job: dict[LLMInstantiationType, hub.client.CompileJob] = field(
        default_factory=dict
    )
    subcomponent_profile_job: dict[LLMInstantiationType, hub.client.ProfileJob] = field(
        default_factory=dict
    )
    subcomponent_inference_job: dict[LLMInstantiationType, hub.client.InferenceJob] = (
        field(default_factory=dict)
    )

    def name(self, num_components: int) -> str:
        """Name for referring to this component."""
        return f"part_{self.component_idx + 1}_of_{num_components}"

    def subcomponent_name(
        self, instantiation_type: LLMInstantiationType, num_components: int
    ) -> str:
        """Name for referring to subcomponent corresponding to this component + instantiation_type."""
        return f"{instantiation_type.value}_{self.name(num_components)}"


__all__ = [
    "LLMInstantiationType",
    "LLMInstantiation",
    "LLMComponent",
]
