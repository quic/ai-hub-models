# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image

from qai_hub_models.models._shared.cvt_gkt.app import CVT_GKTApp


class CVTApp(CVT_GKTApp):
    """
    Lightweight application wrapper for Cross-View Transformers.

    For a given set of images, the app will:
        * Pre-process the inputs (convert to range [0, 1] and apply transformations)
        * Run the inference
        * Post-process BEV heatmap to generate map-view images
        * Return the resulting images with heatmap and ego polygon
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ckpt_name: str,
        target_height: int = 224,
        target_width: int = 480,
    ) -> None:
        """
        Initialize CVTApp.

        Parameters
        ----------
        model
            Cross-View Transformer model accepting a dictionary input and returning a dictionary.
        ckpt_name
            Checkpoint name (e.g., "vehicles_50k" or "road_75k"). Default is CKPT_NAME.
        target_height
            Target height for input images. If None, uses model's default height.
        target_width
            Target width for input images. If None, uses model's default width.
        """
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
            Each camera dictionary contains:
                - intrins : np.ndarray, shape (3, 3)
                    Camera intrinsics matrix.
                - sensor2ego_translation : np.ndarray, shape (3,)
                    Translation vector [x, y, z] in meters from sensor to ego frame.
                - sensor2ego_rotation : np.ndarray, shape (4,)
                    Quaternion [w, x, y, z] for rotation from sensor to ego frame.
                - ego2global_translation : np.ndarray, shape (3,)
                    Translation vector [x, y, z] in meters from ego to global frame.
                - ego2global_rotation : np.ndarray, shape (4,)
                    Quaternion [w, x, y, z] for rotation from ego to global frame.
        raw_output
            If True, return raw BEV heatmap tensor. If False, return processed RGB images.
            Default is False.

        Returns
        -------
        result : list[Image.Image] | torch.Tensor
            If raw_output is False:
                list of PIL Images (RGB) with heatmap and ego polygon.
            If raw_output is True:
                BEV heatmap tensor with shape [1, 1, 200, 200].
        """
        images_tensor, _, _, inv_intrinsics_tensor, inv_extrinsics_tensor = (
            self.preprocess_images(images, cam_metadata)
        )
        bev = self.model(images_tensor, inv_intrinsics_tensor, inv_extrinsics_tensor)

        if raw_output:
            return bev
        return self.postprocess_bev_to_image(bev)
