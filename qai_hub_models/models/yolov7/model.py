# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from importlib import reload
from typing import Any, List, Mapping, Optional

import torch

from qai_hub_models.models._shared.yolo.utils import (
    detect_postprocess,
    yolo_sample_inputs,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel, InputsType
from qai_hub_models.utils.input_spec import InputSpec

YOLOV7_SOURCE_REPOSITORY = "https://github.com/WongKinYiu/yolov7"
YOLOV7_SOURCE_REPO_COMMIT = "84932d70fb9e2932d0a70e4a1f02a1d6dd1dd6ca"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolov7-tiny.pt"
MODEL_ASSET_VERSION = 1


class YoloV7(BaseModel):
    """Exportable YoloV7 bounding box detector, end-to-end."""

    def __init__(
        self,
        yolov7_feature_extractor: torch.nn.Module,
        yolov7_detector: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.yolov7_feature_extractor = yolov7_feature_extractor
        self.yolov7_detector = yolov7_detector

    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    @classmethod
    def from_pretrained(
        cls,
        weights_name: Optional[str] = DEFAULT_WEIGHTS,
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
        detector_head_state_dict["f"] = yolov7_model.model[
            -1
        ].f  # Previous (input) node indices in sequential model
        detector_head_state_dict["i"] = yolov7_model.model[
            -1
        ].i  # Index in sequential model
        yolov7_detect = _YoloV7Detector.from_yolov7_state_dict(detector_head_state_dict)

        return cls(
            yolov7_model,
            yolov7_detect,
        )

    def forward(self, image: torch.Tensor):
        """
        Run YoloV7 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            boxes: Shape [batch, num preds, 4] where 4 == (center_x, center_y, w, h)
            class scores multiplied by confidence: Shape [batch, num_preds, # of classes (typically 80)]
        """
        feature_extraction_output = (
            *self.yolov7_feature_extractor(image),
        )  # Convert output list to Tuple, for exportability
        prediction = self.yolov7_detector(feature_extraction_output)
        return detect_postprocess(prediction)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, num_channels, height, width), "float32")}

    def sample_inputs(self, input_spec: InputSpec | None = None) -> InputsType:
        if input_spec is not None and input_spec != YoloV7.get_input_spec():
            raise ValueError("Sample input has a fixed size that cannot be changed")

        return yolo_sample_inputs()


class _YoloV7Detector(torch.nn.Module):  # YoloV7 Detection
    """Converts features extracted by YoloV7 to predicted bounding boxes & associated class predictions."""

    def __init__(
        self,
        stride: torch.Tensor,
        f,
        i,
        num_anchors: int,
        num_layers: int,
        m_in_channels: List[int],
        m_out_channel,
    ):
        super(_YoloV7Detector, self).__init__()
        self.f = f
        self.i = i
        self.stride = stride
        self.na = num_anchors
        self.no = m_out_channel // self.na  # number of outputs per anchor
        self.nc = self.no - 5  # number of classes
        self.nl = num_layers
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

        out = _YoloV7Detector(
            state_dict["stride"],
            state_dict["f"],
            state_dict["i"],
            na,
            nl,
            m_in_channels,
            m_out_channel,
        )
        out.load_state_dict(new_state_dict, strict)
        return out

    def make_grid_points(self, x, i):
        x = x.sigmoid()
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        grid = self._make_grid(nx, ny)
        y = x
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * self.stride[i]
        wh = (y[..., 2:4] * 2) ** 2 * self.__getattr__(f"anchor_grid_{i}")

        cat = torch.cat((xy, wh, y[..., 4:]), -1)
        return cat.view(bs, -1, self.no)

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
        z = []  # inference output
        for i in range(self.nl):
            x = all_x[i]
            x = self.m[i](x)  # conv
            points = self.make_grid_points(x, i)
            z.append(points)

        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
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
    ):
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
