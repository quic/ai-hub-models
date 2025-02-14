# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.printing import print_mmcv_import_failure_and_exit

try:
    from mmpose.apis import MMPoseInferencer
except ImportError as e:
    print_mmcv_import_failure_and_exit(e, "rtmpose_body2d", "MMPose")

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_INFERENCER_ARCH = "rtmpose-m_8xb64-270e_coco-wholebody-256x192"
# More inferencer architectures for RTMPose can be found here
# https://github.com/open-mmlab/mmpose/tree/configs/RTMPose/body_2d_keypoint/topdown_heatmap/coco

SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_rtmpose_inputs.npy"
)


class RTMPosebody2d(BaseModel):
    """Exportable RTMPose body2d detector, end-to-end."""

    def __init__(self, inferencer) -> None:
        super().__init__()

        self.inferencer = inferencer
        self.model = self.inferencer.inferencer.model
        self.pre_processor = self.inferencer.inferencer.model.data_preprocessor

    @classmethod
    def from_pretrained(cls) -> RTMPosebody2d:
        """RTMPose comes from the MMPose library, so we load using an internal config
        rather than a public weights file"""
        inferencer = MMPoseInferencer(
            DEFAULT_INFERENCER_ARCH, device=torch.device(type="cpu")
        )
        return cls(inferencer)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Forward pass for processing the inout image ad obtaining the model outputs.
            Args :
            -image (torch.Tensor) : Input tensor of shape (N, C, H, W)
        .
            Returns:
            -tuple[torch.Tensor, torch.Tensor]: A tuple containg:
                -output 1: SimCC x- axis predictions with shaoe (N, 17, 384), where :
                    N = batch size ,
                    133 = Number of keypoints,
                    384 = SimCC X-axis resolution.
                -Output 2: Simcc Y-axis predictions with shape (N, 17, 512), where:
                    N = batch size ,
                    133 = Number of keypoints,
                    512 = SimCC Y-axis resolution.
        """

        x = image[:, [2, 1, 0], ...]  # RGB -> BGR
        x = (x - self.pre_processor.mean) / self.pre_processor.std
        return self.model._forward(x)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [load_numpy(SAMPLE_INPUTS)]}

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 256,
        width: int = 192,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["pred_x", "pred_y"]
