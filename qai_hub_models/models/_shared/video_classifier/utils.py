# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
import torchvision.io
from torchvision import transforms

from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


def normalize(video: torch.Tensor):
    """Normalize the video frames.
    Parameters:
        video: Video tensor (Number of frames x HWC) with values between 0-255
               Channel Layout: RGB

    Returns:
        video: Video is normalized to have values between 0-1
               and transposed so the shape is Channel x Number of frames x HW.
    """
    return video.permute(3, 0, 1, 2).to(torch.float32) / 255


def sample_video(video: torch.Tensor, num_frames: int):
    """
    Samples the number of frames in the video to the number requested.

    Parameters:
        video: A [B, C, Number of frames, H, W] video.
        num_frames: Number of frames to sample video down to.

    Returns:
        video: Video tensor sampled to the appropriate number of frames.
    """
    frame_rate = video.shape[0] // num_frames
    return video[: frame_rate * num_frames : frame_rate]


def resize(video: torch.Tensor, size: tuple[int, int]):
    """
    Interpolate the frames of the image to match model's input resolution.

    Parameters:
        video: torch.Tensor

    Returns:
        video: Resized video is returned.
               Selected settings for resize were recommended.

    """
    return torch.nn.functional.interpolate(
        video, size=size, scale_factor=None, mode="bilinear", align_corners=False
    )


def crop(video: torch.Tensor, output_size: tuple[int, int]):
    """
    Parameters:
        video: torch.Tensor
            Input video torch.Tensor.
        output_size: desired output shape for each frame.

    Returns:
        video: torch.Tensor
            Center cropped based on the output size

    """
    h, w = video.shape[-2:]
    th, tw = output_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return video[..., i : (i + th), j : (j + tw)]


def normalize_base(
    video: torch.Tensor, mean: list[float], std: list[float]
) -> torch.Tensor:
    """

    Parameters:
        video: Input video torch.Tensor
        mean: Mean to be subtracted per channel of the input.
        std: Standard deviation to be divided per each channel.

    Returns:
        video: Normalized based on provided mean and scale.
               The operation is done per channel.

    """
    shape = (-1,) + (1,) * (video.dim() - 1)
    mean_tensor = torch.as_tensor(mean).reshape(shape)
    std_tensor = torch.as_tensor(std).reshape(shape)
    return (video - mean_tensor) / std_tensor


def read_video_per_second(path: str) -> torch.Tensor:
    """

    Parameters:
        path: Path of the input video.

    Returns:
        input_video: Reads video from path and converts to torch tensor.

    """
    input_video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    return input_video


def preprocess_video_kinetics_400(input_video: torch.Tensor):
    """
    Preprocess the input video correctly for video classification inference.

    This is specific to torchvision models that take input of size 112.

    Sourced from: https://github.com/pytorch/vision/tree/main/references/video_classification

    Parameters:
        input_video: Raw input tensor

    Returns:
        video: Normalized, resized, cropped and normalized by channel for input model.
    """
    input_video = normalize(input_video)
    input_video = resize(input_video, (128, 171))
    input_video = crop(input_video, (112, 112))
    return input_video


def preprocess_video_224(input_video: torch.Tensor):
    """
    Preprocess the input video correctly for video classification inference.

    This is specific to models like video_mae which take inputs of size 224.

    Sourced from: https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/kinetics.py#L56

    Parameters:
        input_video: Raw input tensor

    Returns:
        video: Normalized, resized, cropped and normalized by channel for input model.
    """
    input_video = normalize(input_video)
    transform = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.CenterCrop((224, 224)),
        ]
    )
    return transform(input_video)


def get_class_name_kinetics_400() -> list[str]:
    """
    Return the list of class names in the correct order, where the class index
    within this list corresponds to logit at the same index of the model output.
    """
    labels_path = QAIHM_PACKAGE_ROOT / "labels" / "kinetics400_labels.txt"
    with open(labels_path) as f:
        return [line.strip() for line in f.readlines()]
