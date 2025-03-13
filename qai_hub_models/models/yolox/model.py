# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.models.yolov3.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov3_demo_640.jpg"
)

SOURCE_REPOSITORY = "https://github.com/Megvii-BaseDetection/YOLOX"
SOURCE_REPO_COMMIT = "d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolox_m.pth"
MODEL_ASSET_VERSION = 1


class YoloX(Yolo):
    """Exportable yolox-m bounding box detector, end-to-end."""

    def __init__(
        self,
        model: torch.nn.Module,
        include_postprocessing: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ):
        """Load yolox-m from a weightfile created by the source Yolox repository."""
        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, weights_name
        ).fetch()
        # Load PyTorch model from disk
        yolox_model = _load_yolox_source_model_from_weights(checkpoint_path)
        return cls(yolox_model, include_postprocessing)

    def forward(self, image):
        """
        Run YoloX on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float
                   3-channel Color Space: RGB [0-1]

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    Confidence score that the given box is the predicted class: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            else:
                predictions: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)

        """
        # Scale the image pixel values from [0, 1] to [0, 255] as per yolox requirement
        image = image * 255
        predictions = self.model(image)
        boxes, scores, class_idx = detect_postprocess(predictions)
        return boxes, scores, class_idx if self.include_postprocessing else predictions

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.include_postprocessing)


def _load_yolox_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load Yolox model from the source repository using the given weights.
    # Returns <source repository>.models.yolo.Model
    with SourceAsRoot(
        SOURCE_REPOSITORY,
        SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):

        from torch import nn
        from yolox.exp import get_exp
        from yolox.models.network_blocks import SiLU
        from yolox.utils import replace_module

        exp = get_exp(exp_name="yolox-m")
        model = exp.get_model()
        ckpt = torch.load(weights_name, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        # pytorch SiLU activation function has to be replaced by a custom implementation
        model = replace_module(model, nn.SiLU, SiLU)
        # enable decoding in the model head
        model.head.decode_in_inference = True
        model.to("cpu").eval()
        return model
