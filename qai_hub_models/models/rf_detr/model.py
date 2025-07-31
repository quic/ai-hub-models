# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.detr.model import DETR
from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

# TODO: Temporarily using GitHub repository for installation. Switch back to PyPI-based install once the development code is released.
# https://github.com/qcom-ai-hub/ai-hub-models-internal/issues/1634
SOURCE_REPOSITORY = "https://github.com/roboflow/rf-detr.git"
SOURCE_REPO_COMMIT = "1e63dbad402eea10f110e86013361d6b02ee0c09"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "rfdetr-base"
MODEL_ASSET_VERSION = 1

DEFAULT_RESOLUTION = 560


class RF_DETR(DETR):
    """Exportable RF-DETR model, end-to-end."""

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run RF-DETR on `image` and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.

        Returns:
            A tuple of three tensors:
                - boxes: torch.Tensor of shape (1, 100, 4) representing the bounding box coordinates (x1, y1, x2, y2)
                - scores: torch.Tensor of shape (1, 100) representing the confidence scores
                - labels: torch.Tensor of shape (1, 100) representing the class labels
        """
        image_array = normalize_image_torchvision(image)
        # boxes: (center_x, center_y, w, h)
        predictions = self.model(image_array)
        # RF-DETR has swapped output order compared to standard DETR
        # logits are at index 1, boxes are at index 0
        logits, boxes = predictions[1], predictions[0]
        boxes, scores, labels = self.detr_postprocess(logits, boxes, image_array.shape)

        return boxes, scores, labels

    @classmethod
    def from_pretrained(cls, weights: str | None = None):
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            from rfdetr import RFDETRBase

            torch_model = RFDETRBase(resolution=DEFAULT_RESOLUTION, device="cpu")
            torch_model.optimize_for_inference(compile=False)  # type: ignore
            return cls(torch_model.model.inference_model)  # type: ignore

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = DEFAULT_RESOLUTION,
        width: int = DEFAULT_RESOLUTION,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return DETR.get_input_spec(batch_size, height, width)
