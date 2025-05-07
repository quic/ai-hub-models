# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.transform import resize

from qai_hub_models.models.fomm.model import FOMM
from qai_hub_models.utils.image_processing import pad_to_square

EXPECTED_SIZE = 256


def resize_frame(
    frame: np.ndarray, size: tuple[int, int] = (EXPECTED_SIZE, EXPECTED_SIZE)
) -> np.ndarray:
    return resize(pad_to_square(frame), size)[..., :3]


# This comes from https://github.com/AliaksandrSiarohin/first-order-model/blob/master/animate.py
# We copy it here to use between the keypoint detector and the generator,
# so it is not baked into the models by tracing.
# The only modification is instead of inputs being dictionaries with keys "value" and "jacobian",
# these are split into separate variables
def normalize_kp(
    kp_source_value,
    kp_source_jacobian,
    kp_driving_value,
    kp_driving_jacobian,
    kp_driving_initial_value,
    kp_driving_initial_jacobian,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source_value[0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial_value[0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {"value": kp_driving_value, "jacobian": kp_driving_jacobian}

    if use_relative_movement:
        kp_value_diff = kp_driving_value - kp_driving_initial_value
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source_value

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving_jacobian, torch.inverse(kp_driving_initial_jacobian)
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source_jacobian)

    return kp_new


def load_video_frames(
    video_filepath: str | Path, sample_rate: int = 1
) -> list[np.ndarray]:
    """
    Loads video frames from a .mp4 file.

    Parameters:
        video_filepath: Filepath to the .mp4 file.
        sample_rate: Only return one every {sample_rate} frames.
            By default, returns all frames.

    Returns:
        List of frames a numpy arrays.
    """
    try:
        reader = imageio.get_reader(video_filepath)
    except RuntimeError as e:
        if "ffmpeg" in str(e):
            raise ValueError(
                "This demo requires the ffmpeg package to be installed on the system. "
                "For example, on a mac, you can run `brew install ffmpeg`."
            )
        raise

    # Drop any misbehaving frames
    driving_video_frames = []
    i = 0
    try:
        for im in reader.iter_data():
            curr_frame = resize_frame(im)
            if i % sample_rate == 0:
                driving_video_frames.append(curr_frame)
            i += 1
    except RuntimeError:
        pass
    reader.close()
    return driving_video_frames


class FOMMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FOMM.

    The app uses 1 model:
        * FOMM

    For a given source image and driving video input, the app will:
        * detect keypoints from each frame of the driving video
        * detect keypoints from the source image
        * generate new frames with the source image performing the motion of the driving video
    """

    def __init__(self, model: FOMM):
        self.kp_detector = model.detector
        self.generator = model.generator

    def predict(self, *args, **kwargs):
        # See copy_motion_to_video.
        return self.copy_motion_to_video(*args, **kwargs)

    def copy_motion_to_video(
        self,
        source_image: np.ndarray,
        driving_video: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Given a source image and a driving video, returns a new video of the
            source image mirroring the movements in the driving video.

        Parameters:
            source_image: numpy array (H W C x uint8)

            driving_video: List of numpy array frames of the target animation

        Returns:
            predictions: list[numpy array H W C x float32 [0, 1]]
                A list of predicted images
        """
        # open source image
        source_image = resize_frame(source_image)
        source_frame = torch.tensor(
            source_image[np.newaxis].astype(np.float32)
        ).permute(0, 3, 1, 2)

        driving_tensors = torch.tensor(
            np.array(driving_video)[np.newaxis].astype(np.float32)
        ).permute(0, 4, 1, 2, 3)

        # Get keypoints in source image
        source_kp_v, source_kp_j = self.kp_detector(source_frame)

        # get keypoints in initial target frame
        driving_kp_initial_v, driving_kp_initial_j = self.kp_detector(
            driving_tensors[:, :, 0]
        )
        predictions = []
        for frame_idx in range(driving_tensors.shape[2]):
            driving_frame = driving_tensors[:, :, frame_idx]
            # get keypoints for each frame of driving video
            driving_kp_v, driving_kp_j = self.kp_detector(driving_frame)

            # normalize the keypoints
            kp_norm = normalize_kp(
                kp_source_value=source_kp_v,
                kp_source_jacobian=source_kp_j,
                kp_driving_value=driving_kp_v,
                kp_driving_jacobian=driving_kp_j,
                kp_driving_initial_value=driving_kp_initial_v,
                kp_driving_initial_jacobian=driving_kp_initial_j,
                use_relative_jacobian=True,
                adapt_movement_scale=True,
            )
            # predict new frame
            out = self.generator(
                source_frame,
                source_kp_v,
                source_kp_j,
                kp_norm["value"],
                kp_norm["jacobian"],
            )

            predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])

        return predictions
