# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from functools import partial
from typing import cast

import torch

from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.input_spec import InputSpec

# ContextManager for running code with MediaPipePyTorch in python path and the
# root directory of MediaPipePyTorch set as cwd
MediaPipePyTorchAsRoot = partial(
    SourceAsRoot,
    "https://github.com/zmurez/MediaPipePyTorch",
    "65f2549ba35cd61dfd29f402f6c21882a32fabb1",
    "mediapipe_pytorch",
    1,
)


def trace_mediapipe(
    detector_input_spec: InputSpec,
    box_detector: torch.nn.Module,
    landmark_input_spec: InputSpec,
    landmark_detector: torch.nn.Module,
) -> tuple[torch.ScriptModule, torch.ScriptModule]:
    # Convert the models to pytorch traces. Traces can be saved & loaded from disk.
    # With Qualcomm® AI Hub, a pytorch trace can be exported to run efficiently on mobile devices!
    #
    # Returns: tuple[Box Detector Trace Object, Landmark Detector Trace Object]
    #
    box_detector_input_shape = detector_input_spec["image"][0]
    box_detector_trace = torch.jit.trace(
        box_detector, [torch.rand(box_detector_input_shape)]
    )

    landmark_detector_input_shape = landmark_input_spec["image"][0]
    landmark_detector_trace = torch.jit.trace(
        landmark_detector, [torch.rand(landmark_detector_input_shape)]
    )

    return cast(torch.ScriptModule, box_detector_trace), cast(
        torch.ScriptModule, landmark_detector_trace
    )


def decode_preds_from_anchors(
    box_coords: torch.Tensor, img_size: tuple[int, int], anchors: torch.Tensor
) -> None:
    """
    Decode predictions using the provided anchors.

    This function can be exported and run inside inference frameworks if desired.

    Note: If included in the model, this code is likely to be unfriendly to quantization.
          This is because of the high range and variability of the output tensor.

          For best quantization accuracy, this code should be run separately from the model,
          or the model should de-quantize activations before running these layers.

    Inputs:
        box_coords: torch.Tensor
            coordinates. Range must be [0, 1]. Shape is [Batch, Num Anchors, 2, 2]
            where [2, 2] == [[xcenter, ycenter], [w, h]]

        img_size: tuple(int, int)
            The size of the tensor that was fed to the NETWORK (NOT the original image size).
            H / W is the same order as coordinates.

        anchors: float
            box anchors. Range must be [0, 1]. Shape is [Batch, Num Anchors, 2, 2],
            where [2, 2] == [[xcenter, ycenter], [w, h]]

        pad: tuple(int, int)
            Padding used during resizing of input image to network input tensor. (w, h)
            This is the absolute # of padding pixels in the network input tensor, NOT in the original image.

    Outputs:
        coordinates: [..., m] tensor, where m is always (x0, y0)
            The absolute coordinates of the box in the original image.
            The "coordinates" input is modified in place.
    """
    assert box_coords.shape[-1] == anchors.shape[-1] == 2
    assert box_coords.shape[-3] == anchors.shape[-3]

    w_size, h_size = img_size
    anchors_x, anchors_y, anchors_w, anchors_h = (
        anchors[..., 0, 0],
        anchors[..., 0, 1],
        anchors[..., 1, 0],
        anchors[..., 1, 1],
    )
    expanded_anchors_shape = [*list(anchors_w.shape), 1]

    # Determine real center X and Y, as well as real pixel W and H
    box_coords[..., 0, 0] = (
        box_coords[..., 0, 0] / w_size * anchors_w + anchors_x
    )  # x_center
    box_coords[..., 0, 1] = (
        box_coords[..., 0, 1] / h_size * anchors_h + anchors_y
    )  # y_center
    box_coords[..., 1, 0] = box_coords[..., 1, 0] / w_size * anchors_w  # w
    box_coords[..., 1, 1] = box_coords[..., 1, 1] / h_size * anchors_h  # h

    # Get X and Y values of keypoints
    box_coords[..., 2:, 0] = box_coords[..., 2:, 0] / w_size * anchors_w.view(
        expanded_anchors_shape
    ) + anchors_x.view(expanded_anchors_shape)
    box_coords[..., 2:, 1] = box_coords[..., 2:, 1] / h_size * anchors_h.view(
        expanded_anchors_shape
    ) + anchors_y.view(expanded_anchors_shape)


def preprocess_hand_x64(
    pts: torch.Tensor, handedness: torch.Tensor, mirror: bool = False
) -> torch.Tensor:
    """
    Normalize hand landmarks, flatten (63), and concatenate handedness (1) → x64.

    pts: (N, 21, 3), handedness: (N, 1) , mirror: (True/False)
    Returns: x64 tensor of shape (N, 64)

    Notes
    -----
    - This preprocessing is performed outside the model due to w8a8 quantization accuracy
    """
    if mirror:
        x_mirror = torch.tensor([-1.0, 1.0, 1.0]).view(1, 1, 3)
        pts = pts * x_mirror
        handedness = 1.0 - handedness
    # Fixed normalization configuration
    center_idx = torch.tensor(
        [0, 1, 5, 9, 13, 17], dtype=torch.long
    )  # stable anatomical anchors
    epsilon = 1e-5  # small constant to avoid divide-by-zero

    # Compute center from selected landmarks
    center = pts[:, center_idx, :].mean(dim=1, keepdim=True)  # (N, 1, 3)

    # Translate points so center is at origin
    normed = pts - center

    # Compute scale based on max range in X or Y
    x = normed[..., 0]
    y = normed[..., 1]
    range_x = x.max(dim=1).values - x.min(dim=1).values
    range_y = y.max(dim=1).values - y.min(dim=1).values
    scale = torch.maximum(range_x, range_y).view(-1, 1, 1) + epsilon

    # Normalize and flatten
    pts_n = normed / scale
    flat = pts_n.reshape(pts_n.shape[0], 63)

    # Append handedness scalar
    return torch.cat([flat, handedness.view(-1, 1).float()], dim=1)
