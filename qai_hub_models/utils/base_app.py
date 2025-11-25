# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

RUN_MODEL_RETURN_TYPE = list[torch.Tensor] | torch.Tensor
CollectionAppTypeVar = TypeVar("CollectionAppTypeVar", bound="CollectionAppProtocol")


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


@runtime_checkable
class CollectionAppProtocol(Protocol):
    """Method required to get calibration data for CollectionModels."""

    @classmethod
    def get_calibration_data(
        cls,
        model: BaseModel,
        calibration_dataset_name: str,
        num_samples: int | None,
        input_spec: InputSpec,
        collection_model: CollectionModel,
    ) -> DatasetEntries:
        """
        Produces a numpy dataset to be used for calibration data of a quantize job.

        Parameters
        ----------
            model: The model for which to get calibration data.
            calibration_dataset_name: Dataset name to use for calibration.
            num_samples: Number of data samples to use. If not specified, uses
                default specified on dataset.
            input_spec: The input spec of the model. Used to ensure the returned
                dataset's names match the input names of the model.
            collection_model: It is required when using app-based calibration.

        Returns
        -------
            Dataset compatible with the format expected by AI Hub.
        """
        ...
