# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch

from qai_hub_models.models._shared.facemap_3dmm.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy


class FaceMap_3DMMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceMap_3DMMApp.

    The app uses 1 model:
        * FaceMap_3DMM

    For a given image input, the app will:
        * pre-process the image (convert to range[-1, 1])
        * Run FaceMap_3DMM inference
        * Convert the output parameters into landmark points
        * Display the output lanrmarks on the image
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

        # 3DMM related parameters
        face_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "meanFace.npy"
        )

        basis_id_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "shapeBasis.npy"
        )

        basis_exp_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "blendShape.npy"
        )

        self.face = torch.from_numpy(load_numpy(face_path).reshape(3 * 68, 1))
        self.basis_id = torch.from_numpy(load_numpy(basis_id_path).reshape(3 * 68, 219))
        self.basis_exp = torch.from_numpy(
            load_numpy(basis_exp_path).reshape(3 * 68, 39)
        )
        self.vertex_num = 68
        self.alpha_id_size = 219
        self.alpha_exp_Size = 39

    def predict(self, *args, **kwargs):
        return self.landmark_prediction(*args, **kwargs)

    def landmark_prediction(
        self,
        _image: np.ndarray,
        x0: np.int32,
        x1: np.int32,
        y0: np.int32,
        y1: np.int32,
    ) -> tuple:
        """
        Return the input image with the predicted lmk overlayed on it.

        Parameters:
            _image: numpy array (H W C x uint8) -- RGB channel layout
            x0: numpy int32 -- left coordinate of face bounding box
            x1: numpy int32 -- right coordinate of face bounding box
            y0: numpy int32 -- top coordinate of face bounding box
            y1: numpy int32 -- bottom coordinate of face bounding box

        Returns:
            lmk_images: numpy array -- images with predicted landmarks displayed.
        """
        height = y1 - y0 + 1
        width = x1 - x0 + 1

        CHW_fp32_torch_crop_image = torch.from_numpy(
            cv2.resize(
                _image[y0 : y1 + 1, x0 : x1 + 1],
                (128, 128),
                interpolation=cv2.INTER_LINEAR,
            )
        ).float()

        output = self.model(
            CHW_fp32_torch_crop_image.permute(2, 0, 1).view(1, 3, 128, 128)
        )

        # Parse results from network
        alpha_id, alpha_exp, pitch, yaw, roll, tX, tY, f = (
            output[0, 0:219],
            output[0, 219:258],
            output[0, 258],
            output[0, 259],
            output[0, 260],
            output[0, 261],
            output[0, 262],
            output[0, 263],
        )

        # De-normalized to original range from [-1, 1]
        alpha_id = alpha_id * 3
        alpha_exp = alpha_exp * 0.5 + 0.5
        pitch = pitch * np.pi / 2
        yaw = yaw * np.pi / 2
        roll = roll * np.pi / 2
        tX = tX * 60
        tY = tY * 60
        tZ = 500
        f = f * 150 + 450

        p_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(-torch.tensor(np.pi)), -torch.sin(-torch.tensor(np.pi))],
                [0, torch.sin(-torch.tensor(np.pi)), torch.cos(-torch.tensor(np.pi))],
            ]
        )

        # Create a rotation matrix from pitch, yaw, roll
        roll_matrix = torch.tensor(
            [
                [torch.cos(-roll), -torch.sin(-roll), 0],
                [torch.sin(-roll), torch.cos(-roll), 0],
                [0, 0, 1],
            ]
        )

        yaw_matrix = torch.tensor(
            [
                [torch.cos(-yaw), 0, torch.sin(-yaw)],
                [0, 1, 0],
                [-torch.sin(-yaw), 0, torch.cos(-yaw)],
            ]
        )

        pitch_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(-pitch), -torch.sin(-pitch)],
                [0, torch.sin(-pitch), torch.cos(-pitch)],
            ]
        )

        r_matrix = torch.mm(
            yaw_matrix, torch.mm(pitch_matrix, torch.mm(p_matrix, roll_matrix))
        )

        # Reconstruct face
        vertices = torch.mm(
            (
                self.face
                + torch.mm(self.basis_id, alpha_id.view(219, 1))
                + torch.mm(self.basis_exp, alpha_exp.view(39, 1))
            ).view([self.vertex_num, 3]),
            r_matrix.transpose(0, 1),
        )

        # Apply translation
        vertices[:, 0] += tX
        vertices[:, 1] += tY
        vertices[:, 2] += tZ

        # Project landmark vertices to 2D
        f = torch.tensor([f, f]).float()
        landmark = vertices[:, 0:2] * f / tZ + 128 / 2

        landmark[:, 0] = landmark[:, 0] * width / 128 + x0
        landmark[:, 1] = landmark[:, 1] * height / 128 + y0

        # Draw landmark
        output_image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
        for n in range(landmark.shape[0]):
            output_image = cv2.circle(
                output_image,
                (int(landmark[n, 0]), int(landmark[n, 1])),
                2,
                (0, 0, 255),
                -1,
            )

        return landmark, output_image
