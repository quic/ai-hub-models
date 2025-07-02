# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import torch

from qai_hub_models.utils.base_model import CollectionModel

RUN_MODEL_RETURN_TYPE = Union[list[torch.Tensor], torch.Tensor]


class BaseCollectionApp(ABC):
    @abstractmethod
    def run_model(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[RUN_MODEL_RETURN_TYPE, ...] | RUN_MODEL_RETURN_TYPE:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model: CollectionModel) -> BaseCollectionApp:
        pass
