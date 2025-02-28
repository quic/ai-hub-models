# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Mapping
from importlib import reload
from typing import Any

import torch
import torch.nn.functional as F

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess_split_input
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo

YOLOV7_SOURCE_REPOSITORY = "https://github.com/WongKinYiu/yolov7"
YOLOV7_SOURCE_REPO_COMMIT = "84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolov7-tiny.pt"
MODEL_ASSET_VERSION = 1


class YoloV7(Yolo):
    """Exportable YoloV7 bounding box detector, end-to-end."""

    def __init__(
        self,
        yolov7_feature_extractor: torch.nn.Module,
        yolov7_detector: torch.nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
        class_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.yolov7_feature_extractor = yolov7_feature_extractor
        self.yolov7_detector = yolov7_detector
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        self.class_dtype = class_dtype

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ):
        """Load YoloV7 from a weightfile created by the source YoloV7 repository."""
        # Load PyTorch model from disk
        yolov7_model = _load_yolov7_source_model_from_weights(weights_name)

        yolov7_model.profile = False

        # When traced = True, the model will skip the "Detect" step,
        # which allows us to override it with an exportable version.
        yolov7_model.traced = True

        # Generate replacement detector that can be traced
        detector_head_state_dict = yolov7_model.model[-1].state_dict()
        detector_head_state_dict["stride"] = yolov7_model.model[-1].stride

        h, w = cls.get_input_spec()["image"][0][2:]
        detector_head_state_dict["h"] = h
        detector_head_state_dict["w"] = w
        yolov7_detect = _YoloV7Detector.from_yolov7_state_dict(detector_head_state_dict)

        return cls(
            yolov7_model,
            yolov7_detect,
            include_postprocessing,
            split_output,
        )

    def forward(self, image):
        """
        Run YoloV7 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            else if self.split_output:
                output_xy: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [x_center, y_center] (box_coordinates)

                output_wh: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [width, height] (box_size)

                output_scores: torch.Tensor
                    Shape is [batch, num_preds, j]
                        where j is [confidence (1 element) , # of classes elements]


            else:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, k]
                        where, k = # of classes + 5
                        k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                        and box_coordinates are [x_center, y_center, w, h]
        """
        feature_extraction_output = (
            *self.yolov7_feature_extractor(image),
        )  # Convert output list to Tuple, for exportability
        detector_output = self.yolov7_detector(feature_extraction_output)

        if not self.include_postprocessing:
            if self.split_output:
                return detector_output
            return torch.cat(detector_output, -1)

        return detect_postprocess_split_input(  # type: ignore[misc]
            *detector_output, class_dtype=self.class_dtype
        )

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True, split_output: bool = False
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        if split_output:
            return ["boxes_xy", "boxes_wh", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing, self.split_output
        )


class _YoloV7Detector(torch.nn.Module):  # YoloV7 Detection
    """Converts features extracted by YoloV7 to predicted bounding boxes & associated class predictions."""

    def __init__(
        self,
        stride: torch.Tensor,
        num_anchors: int,
        num_layers: int,
        m_in_channels: list[int],
        m_out_channel,
        input_shape: tuple[int, int],
    ):
        super().__init__()
        self.stride = stride
        self.na = num_anchors
        self.no = m_out_channel // self.na  # number of outputs per anchor
        self.nc = self.no - 5  # number of classes
        self.nl = num_layers
        self.h, self.w = input_shape
        for i in range(0, self.nl):
            self.register_buffer(
                f"anchor_grid_{i}", torch.zeros(1, self.na, 1, 1, 2)
            )  # nl * [ tensor(shape(1,na,1,1,2)) ]
        self.m = torch.nn.ModuleList(
            torch.nn.Conv2d(m_in_channel, m_out_channel, 1)
            for m_in_channel in m_in_channels
        )  # output conv

    @staticmethod
    def from_yolov7_state_dict(
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ):
        """
        Load this module from a state dict taken from the "Detect" module.
        This module is found in the original YoloV7 source repository (models/common.py::Detect).
        """
        new_state_dict = {}

        # Convert anchor grid buffer from rank 6 to several rank 5 tensors, for export-friendliness.
        anchor_grid = state_dict["anchor_grid"]
        nl = len(anchor_grid)
        na = anchor_grid.shape[2]
        for i in range(0, nl):
            new_state_dict[f"anchor_grid_{i}"] = anchor_grid[i]

        # Copy over `m` layers
        m_in_channels = []
        m_out_channel = 0
        for i in range(0, nl):
            weight = f"m.{i}.weight"
            for x in [weight, f"m.{i}.bias"]:
                new_state_dict[x] = state_dict[x]
            m_in_channels.append(new_state_dict[weight].shape[1])
            m_out_channel = new_state_dict[weight].shape[0]

        input_shape = state_dict["h"], state_dict["w"]

        out = _YoloV7Detector(
            state_dict["stride"],
            na,
            nl,
            m_in_channels,
            m_out_channel,
            input_shape,
        )
        out.load_state_dict(new_state_dict, strict)
        return out

    def make_grid_points(self, x, i):
        x = x.sigmoid()
        # bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        stride = int(self.stride[i])
        nx, ny = self.h // stride, self.w // stride
        x = x.reshape(-1, self.na, self.no, nx, ny).permute(0, 1, 3, 4, 2).contiguous()
        # TODO(13933) Revert once QNN issues with ReduceMax are fixed
        # Pad 1 class up to 2 to get NPU residence
        x = F.pad(x, (0, max(7 - self.no, 0)))
        grid = self._make_grid(nx, ny)
        y = x

        # Fp16 NPU only supports tensor math up to rank 4
        # xy computation suffers from accuracy loss when moved to NPU
        # Only convert wh to rank 4
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
        xy = xy.reshape(-1, self.na * nx * ny, 2)

        wh = y[..., 2:4].reshape(-1, self.na, nx * ny, 2)
        wh = (wh * 2) ** 2 * self.__getattr__(f"anchor_grid_{i}").squeeze(2)
        wh = wh.reshape(-1, self.na * nx * ny, 2)

        # TODO(13933) Revert max operation once QNN issues with ReduceMax are fixed
        scores = y[..., 4:].reshape(-1, self.na * nx * ny, max(3, self.no - 4))
        return xy, wh, scores

    def forward(self, all_x: tuple[torch.Tensor, ...]):
        """
        From the outputs of the feature extraction layers of YoloV7, predict bounding boxes,
        classes, and confidence.

        Parameters:
            all_x: tuple[torch.Tensor]
                Outputs of the feature extraction layers of YoloV7. Typically 3 5D tensors.

        Returns:
            pred: [batch_size, # of predictions, 5 + # of classes]
                Where the rightmost dim contains [center_x, center_y, w, h, confidence score, n per-class scores]
        """
        # inference output
        all_xy = []
        all_wh = []
        all_scores = []
        for i in range(self.nl):
            x = all_x[i]
            x = self.m[i](x)  # conv
            xy, wh, scores = self.make_grid_points(x, i)
            all_xy.append(xy)
            all_wh.append(wh)
            all_scores.append(scores)

        return torch.cat(all_xy, 1), torch.cat(all_wh, 1), torch.cat(all_scores, 1)

    @staticmethod
    def _make_grid(nx: int, ny: int):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def _load_yolov7_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load YoloV7 model from the source repository using the given weights.
    # Returns <source repository>.models.yolo.Model
    with SourceAsRoot(
        YOLOV7_SOURCE_REPOSITORY,
        YOLOV7_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ) as repo_path:
        # We don't touch these flags, and having conditionals in `forward`
        # dependent on inputs to the function makes torch fx Graph creation unhappy.
        find_replace_in_repo(repo_path, "models/yolo.py", "if augment:", "if False:")
        find_replace_in_repo(repo_path, "models/yolo.py", "if profile:", "if False:")

        # Our qai_hub_models/models package may already be loaded and cached
        # as "models" (reproduce by running python -m models.yolov7.demo from
        # models qai_hub_models folder). To make sure it loads the external
        # "models" package, explicitly reload first.
        import models

        reload(models)

        # necessary imports. `models` come from the yolov7 repo.
        from models.experimental import attempt_load
        from models.yolo import Model

        yolov7_model = attempt_load(weights_name, map_location="cpu")  # load FP32 model

        assert isinstance(yolov7_model, Model)
        return yolov7_model
