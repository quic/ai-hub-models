# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


class EyeGazeApp:
    """
    End-to-end application for EyeGaze inference.

    Processes a grayscale eye crop image (160x96) to predict gaze direction. Performs:
    - Preprocessing: Resizes to 160x96, converts to grayscale, equalizes histogram, normalizes.
    - Inference: Uses EyeNet model to predict gaze angles and landmarks.
    - Postprocessing: Adjusts yaw for right eye, visualizes gaze with a red arrow.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> Image.Image | np.ndarray:
        return self.predict_gaze_angle(*args, **kwargs)

    def predict_gaze_angle(
        self, eye_img: np.ndarray, side: str = "left", raw_output: bool = False
    ) -> Image.Image | np.ndarray:
        """
        Run inference on an eye image to predict gaze and return a PIL Image with gaze visualization.

        Parameters
        ----------
        eye_img
            Path to grayscale eye image (PNG/JPG) or numpy array.
        side
            Either 'left' or 'right' eye. Default is 'left'.
        raw_output
            If True, return raw gaze output array. If False, return PIL Image with visualization.

        Returns
        -------
        output_image_or_gaze : Image.Image | np.ndarray
            PIL Image with red gaze arrow visualization, or raw gaze output array if raw_output is True.
        """
        # Load image if path is provided
        if isinstance(eye_img, str):
            eye_img = cv2.imread(eye_img, cv2.IMREAD_GRAYSCALE)
            if eye_img is None:
                raise ValueError(f"Failed to load image: {eye_img}")

        # Preprocess
        proc_img = preprocess_eye_crop(eye_img, side)
        img_tensor = torch.from_numpy(proc_img).unsqueeze(0).float().cpu()

        _, _, gaze_pred = self.model(img_tensor)

        # Postprocess
        gaze_out = gaze_pred.squeeze(0).numpy()
        gaze_out = gaze_out.astype(np.float32)

        if raw_output:
            return gaze_out

        if side == "right":
            gaze_out[1] = -gaze_out[1]

        # Create visualization
        vis = cv2.cvtColor((proc_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        iris_center = np.array([80.0, 48.0], dtype=np.float32)  # center of 160x96
        output_image = draw_gaze(vis, iris_center, gaze_out)
        return Image.fromarray(output_image)


def preprocess_eye_crop(img: np.ndarray, side: str) -> np.ndarray:
    """
    Preprocess an eye crop image for inference.

    Parameters
    ----------
    img
        Input eye image as a numpy array.
    side
        Either 'left' or 'right' eye.

    Returns
    -------
    preprocessed_image : np.ndarray
        Preprocessed image (grayscale, 160x96, normalized).
    """
    img = cv2.resize(img, (160, 96))
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    if side == "right":
        img = np.fliplr(img).copy()
    return img


def draw_gaze(
    image_in: np.ndarray,
    eye_pos: np.ndarray,
    pitchyaw: np.ndarray,
    length: float | None = None,
    thickness: int = 2,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw gaze angle on given image with specified eye positions.

    - Adaptive arrow length based on image size if length is None.
    - Normalizes direction vector so arrow length is consistent regardless of angle magnitude.
    - Keeps the arrow endpoint within image bounds.

    Parameters
    ----------
    image_in
        Input image (grayscale or BGR).
    eye_pos
        Eye position coordinates [x, y].
    pitchyaw
        Gaze angles [pitch, yaw].
    length
        Length of the gaze arrow. If None, computed as 0.35 * min(H, W).
    thickness
        Thickness of the gaze arrow.
    color
        Color of the gaze arrow in RGB format.

    Returns
    -------
    output_image : np.ndarray
        Image with gaze arrow drawn.
    """
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    H, W = image_out.shape[:2]
    # Adaptive length based on image size
    if length is None:
        length = 0.35 * float(min(H, W))

    # 2D direction from pitch/yaw
    dx = -np.sin(float(pitchyaw[1]))
    dy = np.sin(float(pitchyaw[0]))

    # Normalize to unit length to keep arrow length consistent
    norm = np.hypot(dx, dy)
    if norm > 1e-6:
        dx /= norm
        dy /= norm

    dx *= length
    dy *= length

    start_pt = np.round(eye_pos).astype(np.int32)
    end_pt = np.array([eye_pos[0] + dx, eye_pos[1] + dy], dtype=np.float32)

    # Keep endpoint inside image bounds
    end_pt[0] = np.clip(end_pt[0], 0, W - 1)
    end_pt[1] = np.clip(end_pt[1], 0, H - 1)
    end_pt_int = tuple(np.round(end_pt).astype(np.int32))

    cv2.arrowedLine(
        image_out,
        tuple(start_pt),
        end_pt_int,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image_out
