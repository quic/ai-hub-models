# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]


def project_landmark(output):
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

    vertex_num = 68
    alpha_id_size = 219
    alpha_exp_Size = 39

    face = torch.from_numpy(load_numpy(face_path).reshape(3 * vertex_num, 1))
    basis_id = torch.from_numpy(
        load_numpy(basis_id_path).reshape(3 * vertex_num, alpha_id_size)
    )
    basis_exp = torch.from_numpy(
        load_numpy(basis_exp_path).reshape(3 * vertex_num, alpha_exp_Size)
    )

    # Parse results from network
    alpha_id, alpha_exp, pitch, yaw, roll, tX, tY, f = (
        output[0:219],
        output[219:258],
        output[258],
        output[259],
        output[260],
        output[261],
        output[262],
        output[263],
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
            face
            + torch.mm(basis_id, alpha_id.view(alpha_id_size, 1))
            + torch.mm(basis_exp, alpha_exp.view(alpha_exp_Size, 1))
        ).view([vertex_num, 3]),
        r_matrix.transpose(0, 1),
    )

    # Apply translation
    vertices[:, 0] += tX
    vertices[:, 1] += tY
    vertices[:, 2] += tZ

    # Project landmark vertices to 2D
    f = torch.tensor([f, f]).float()
    landmark = vertices[:, 0:2] * f / tZ
    return landmark
