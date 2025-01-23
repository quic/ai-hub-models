# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
import torchvision.io


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
               The operaion is done per channle.

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

    Parameters:
        input_video: Raw input tensor

    Returns:
        video: Normalized, resized, cropped and normalized by channel for input model.
               This preprocessing is dd

    """
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    input_video = normalize(input_video)
    input_video = resize(input_video, (128, 171))
    input_video = crop(input_video, (112, 112))
    input_video = normalize_base(input_video, mean=mean, std=std)
    return input_video
