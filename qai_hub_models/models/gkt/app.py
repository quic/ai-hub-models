# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image

from qai_hub_models.models._shared.cvt_gkt.app import CVT_GKTApp


class GKTApp(CVT_GKTApp):
    """
    Lightweight application wrapper for GKT.

    For a given set of images, the app will:
        * Pre-process the inputs (convert to range [0, 1] and apply transformations)
        * Run the inference
        * Post-process BEV heatmap to generate map-view images
        * Return the resulting images with heatmap and ego polygon
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        ckpt_name: str,
        target_height: int = 224,
        target_width: int = 480,
    ) -> None:
        self.model = model
        super().__init__(
            ckpt_name=ckpt_name, target_height=target_height, target_width=target_width
        )

    def predict_from_images(
        self,
        images: list[Image.Image],
        cam_metadata: dict[str, dict],
        raw_output: bool = False,
    ) -> list[Image.Image] | torch.Tensor:
        """
        Process images and return BEV heatmap or RGB images.

        Parameters
        ----------
        images
            List of 6 RGB PIL images in order: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
            CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT.
        cam_metadata
            Dictionary mapping camera names to camera parameter dictionaries.
            Each camera dictionary contains intrins (shape (3, 3)),
            sensor2ego_translation (shape (3,)), sensor2ego_rotation (shape (4,)),
            ego2global_translation (shape (3,)), and ego2global_rotation (shape (4,)).
        raw_output
            If True, return raw BEV heatmap tensor. If False, return processed RGB images.
            Default is False.

        Returns
        -------
        result
            If raw_output is False, list of PIL Images (RGB) with heatmap and ego polygon.
            If raw_output is True, BEV heatmap tensor with shape [1, 1, 200, 200].
        """
        (
            images_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
            inv_intrinsics_tensor,
            inv_extrinsics_tensor,
        ) = self.preprocess_images(images, cam_metadata)
        bev = self.model(
            images_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
            inv_intrinsics_tensor,
            inv_extrinsics_tensor,
        )

        if raw_output:
            return bev

        return self.postprocess_bev_to_image(bev)
