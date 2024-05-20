# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import math
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from PIL.Image import fromarray as ImageFromArray
from torch.nn.functional import interpolate, pad
from torchvision import transforms

IMAGENET_DIM = 224
IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_DIM),
        transforms.ToTensor(),
    ]
)


def app_to_net_image_inputs(
    pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
) -> Tuple[List[np.ndarray], torch.Tensor]:
    """
    Convert the provided images to application inputs.
    ~~This does not change channel order. RGB stays RGB, BGR stays BGR, etc~~

    Parameters:
        pixel_values_or_image: torch.Tensor
            PIL image
            or
            list of PIL images
            or
            numpy array (H W C x uint8) or (N H W C x uint8) -- both BGR or grayscale channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR or grayscale channel layout

        dst_size: (height, width)
            Size to which the image should be reshaped.

    Returns:
        NHWC_int_numpy_frames: List[numpy.ndarray]
            List of numpy arrays (one per input image with uint8 dtype, [H W C] shape, and BGR or grayscale layout.
            This output is typically used for use of drawing/displaying images with PIL and CV2

        NCHW_fp32_torch_frames: torch.Tensor
            Tensor of images in fp32 (range 0:1), with shape [Batch, Channels, Height, Width], and BGR or grayscale layout.

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    NHWC_int_numpy_frames: List[np.ndarray] = []
    NCHW_fp32_torch_frames: torch.Tensor
    if isinstance(pixel_values_or_image, Image):
        pixel_values_or_image = [pixel_values_or_image]
    if isinstance(pixel_values_or_image, list):
        fp32_frames = []
        for image in pixel_values_or_image:
            NHWC_int_numpy_frames.append(np.array(image.convert("RGB")))
            fp32_frames.append(preprocess_PIL_image(image))
        NCHW_fp32_torch_frames = torch.cat(fp32_frames)
    elif isinstance(pixel_values_or_image, torch.Tensor):
        NCHW_fp32_torch_frames = pixel_values_or_image
        for b_img in pixel_values_or_image:
            NHWC_int_numpy_frames.append((b_img.permute(1, 2, 0) * 255).byte().numpy())
    else:
        assert isinstance(pixel_values_or_image, np.ndarray)
        NHWC_int_numpy_frames = (
            [pixel_values_or_image]
            if len(pixel_values_or_image.shape) == 3
            else [x for x in pixel_values_or_image]
        )
        NCHW_fp32_torch_frames = numpy_image_to_torch(pixel_values_or_image)

    return NHWC_int_numpy_frames, NCHW_fp32_torch_frames


def preprocess_PIL_image(image: Image) -> torch.Tensor:
    """Convert a PIL image into a pyTorch tensor with range [0, 1] and shape NCHW."""
    transform = transforms.Compose([transforms.PILToTensor()])  # bgr image
    img: torch.Tensor = transform(image)  # type: ignore
    img = img.float().unsqueeze(0) / 255.0  # int 0 - 255 to float 0.0 - 1.0
    return img


def preprocess_PIL_image_mask(image_mask: Image) -> torch.Tensor:
    """Convert a PIL mask image into a pyTorch tensor with values 0. or 1."""
    transform = transforms.Compose([transforms.PILToTensor()])
    mask = transform(image_mask.convert("L"))
    mask = mask.unsqueeze(0).float()
    mask = (mask > 1.0) * 1.0
    return mask


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Convert a Numpy image (dtype uint8, shape [H W C] or [N H W C]) into a pyTorch tensor with range [0, 1] and shape NCHW."""
    image_torch = torch.from_numpy(image)
    if len(image.shape) == 3:
        image_torch = image_torch.unsqueeze(0)
    return image_torch.permute(0, 3, 1, 2).float() / 255.0


def torch_tensor_to_PIL_image(data: torch.Tensor) -> Image:
    """
    Convert a Torch tensor (dtype float32) with range [0, 1] and shape CHW into PIL image CHW
    """
    out = torch.clip(data, min=0.0, max=1.0)
    np_out = (out.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    return ImageFromArray(np_out)


def normalize_image_torchvision(
    image_tensor: torch.Tensor, image_tensor_has_batch=True
) -> torch.Tensor:
    """
    Normalizes according to standard torchvision constants.

    Due to issues with FX Graph tracing in AIMET, image_tensor_has_batch is a constant passed in,
    rather than determining the image rank using len(image_tensor.shape).

    There are many PyTorch models that expect input images normalized with
    these specific constants, so this utility can be re-used across many models.
    """
    shape = [-1, 1, 1]
    if image_tensor_has_batch:
        shape.insert(0, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(*shape)
    std = torch.Tensor([[0.229, 0.224, 0.225]]).reshape(*shape)
    return (image_tensor - mean) / std


def normalize_image_transform() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a torchvision transform that returns a torch tensor normalized according to some constants.

    There are many PyTorch models that expect input images normalized with
    these specific constants, so this utility can be re-used across many models.
    """
    return functools.partial(normalize_image_torchvision, image_tensor_has_batch=False)


def pad_to_square(frame: np.ndarray) -> np.ndarray:
    """
    Pad an image or video frame to square dimensions with whitespace.
    Assumes the input shape is of format (H, W, C).
    """
    h, w, _ = frame.shape
    if h < w:
        top_pad = (w - h) // 2
        pad_values = ((top_pad, w - h - top_pad), (0, 0), (0, 0))
    else:
        top_pad = (h - w) // 2
        pad_values = ((0, 0), (top_pad, h - w - top_pad), (0, 0))
    return np.pad(frame, pad_values, constant_values=255)


def resize_pad(image: torch.Tensor, dst_size: Tuple[int, int]):
    """
    Resize and pad image to be shape [..., dst_size[0], dst_size[1]]

    Parameters:
        image: (..., H, W)
            Image to reshape.

        dst_size: (height, width)
            Size to which the image should be reshaped.

    Returns:
        rescaled_padded_image: torch.Tensor (..., dst_size[0], dst_size[1])
        scale: scale factor between original image and dst_size image, (w, h)
        pad: pixels of padding added to the rescaled image: (left_padding, top_padding)

    Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
    """
    height, width = image.shape[-2:]
    dst_frame_height, dst_frame_width = dst_size

    h_ratio = dst_frame_height / height
    w_ratio = dst_frame_width / width
    scale = min(h_ratio, w_ratio)
    if h_ratio < w_ratio:
        scale = h_ratio
        new_height = dst_frame_height
        new_width = math.floor(width * scale)
    else:
        scale = w_ratio
        new_height = math.floor(height * scale)
        new_width = dst_frame_width

    new_height = math.floor(height * scale)
    new_width = math.floor(width * scale)
    pad_h = dst_frame_height - new_height
    pad_w = dst_frame_width - new_width

    pad_top = int(pad_h // 2)
    pad_bottom = int(pad_h // 2 + pad_h % 2)
    pad_left = int(pad_w // 2)
    pad_right = int(pad_w // 2 + pad_w % 2)

    rescaled_image = interpolate(
        image, size=[int(new_height), int(new_width)], mode="bilinear"
    )
    rescaled_padded_image = pad(
        rescaled_image, (pad_left, pad_right, pad_top, pad_bottom)
    )
    padding = (pad_left, pad_top)

    return rescaled_padded_image, scale, padding


def undo_resize_pad(
    image: torch.Tensor,
    orig_size_wh: Tuple[int, int],
    scale: float,
    padding: Tuple[int, int],
):
    """
    Undos the efffect of resize_pad. Instead of scale, the original size
    (in order width, height) is provided to prevent an off-by-one size.
    """
    width, height = orig_size_wh

    rescaled_image = interpolate(image, scale_factor=1 / scale, mode="bilinear")

    scaled_padding = [int(round(padding[0] / scale)), int(round(padding[1] / scale))]

    cropped_image = rescaled_image[
        ...,
        scaled_padding[1] : scaled_padding[1] + height,
        scaled_padding[0] : scaled_padding[0] + width,
    ]

    return cropped_image


def pil_resize_pad(
    image: Image, dst_size: Tuple[int, int]
) -> Tuple[Image, float, Tuple[int, int]]:
    torch_image = preprocess_PIL_image(image)
    torch_out_image, scale, padding = resize_pad(
        torch_image,
        dst_size,
    )
    pil_out_image = torch_tensor_to_PIL_image(torch_out_image[0])
    return (pil_out_image, scale, padding)


def pil_undo_resize_pad(
    image: Image, orig_size_wh: Tuple[int, int], scale: float, padding: Tuple[int, int]
) -> Image:
    torch_image = preprocess_PIL_image(image)
    torch_out_image = undo_resize_pad(torch_image, orig_size_wh, scale, padding)
    pil_out_image = torch_tensor_to_PIL_image(torch_out_image[0])
    return pil_out_image


def denormalize_coordinates(
    coordinates: torch.Tensor,
    input_img_size: Tuple[int, int],
    scale: float = 1.0,
    pad: Tuple[int, int] = (0, 0),
) -> None:
    """
    Maps detection coordinates from [0,1] to coordinates in the original image.

    This function can be exported and run inside inference frameworks if desired.

    Note: If included in the model, this code is likely to be unfriendly to quantization.
          This is because of the high range and variability of the output tensor.

          For best quantization accuracy, this code should be run separately from the model,
          or the model should de-quantize activations before running these layers.

    Inputs:
        coordinates: [..., 2] tensor
            coordinates. Range must be [0, 1]

        input_img_size: Tuple(int, int)
            The size of the tensor that was fed to the NETWORK (NOT the original image size).
            H / W is the same order as coordinates.

        scale: float
            Scale factor that to resize the image to be fed to the network.

        pad: Tuple(int, int)
            Padding used during resizing of input image to network input tensor.
            This is the absolute # of padding pixels in the network input tensor, NOT in the original image.
            H / W is in the same order as coordinates.

    Outputs:
        coordinates: [..., m] tensor, where m is always (y0, x0)
            The absolute coordinates of the box in the original image.
            The "coordinates" input is modified in place.
    """
    img_0, img_1 = input_img_size
    pad_0, pad_1 = pad

    coordinates[..., 0] = ((coordinates[..., 0] * img_0 - pad_0) / scale).int()
    coordinates[..., 1] = ((coordinates[..., 1] * img_1 - pad_1) / scale).int()


def apply_batched_affines_to_frame(
    frame: np.ndarray, affines: List[np.ndarray], output_image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Generate one image per affine applied to the given frame.
    I/O is numpy since this uses cv2 APIs under the hood.

    Inputs:
        frame: np.ndarray
            Frame on which to apply the affine. Shape is [ H W C ], dtype must be np.byte.
        affines: List[np.ndarray]
            List of 2x3 affine matrices to apply to the frame.
        output_image_size: torch.Tensor
            Size of each output frame.

    Outputs:
        images: np.ndarray
            Computed images. Shape is [B H W C]
    """
    assert (
        frame.dtype == np.byte or frame.dtype == np.uint8
    )  # cv2 does not work correctly otherwise. Don't remove this assertion.
    imgs = []
    for affine in affines:
        img = cv2.warpAffine(frame, affine, output_image_size)
        imgs.append(img)
    return np.stack(imgs)


def apply_affine_to_coordinates(
    coordinates: torch.Tensor, affine: torch.Tensor
) -> torch.Tensor:
    """
    Apply the given affine matrix to the given coordinates.

    Inputs:
        coordinates: torch.Tensor
            Coordinates on which to apply the affine. Shape is [ ..., 2 ], where 2 == [X, Y]
        affines: torch.Tensor
            Affine matrix to apply to the coordinates.

    Outputs:
        Transformed coordinates. Shape is [ ..., 2 ], where 2 == [X, Y]
    """
    return (affine[:, :2] @ coordinates.T + affine[:, 2:]).T


def compute_vector_rotation(
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    offset_rads: float | torch.Tensor = 0,
) -> torch.Tensor:
    """
    From the given vector, compute the rotation of the vector with added offset.

    Inputs:
        vec_start: torch.Tensor
            Starting point of the vector. Shape is [B, 2], where 2 == (x, y)
        vec_end: torch.Tensor
            Ending point of the vector. Shape is [B, 2], where 2 == (x, y)
        offset_rads: float | torch.Tensor
            Offset to subtract from the rotation calculation.
            Can be size [1] or [ Batch ]

    Outputs:
        theta: computed rotation angle in radians. Shape is [Batch]
    """
    return (
        torch.atan2(
            vec_start[..., 1] - vec_end[..., 1], vec_start[..., 0] - vec_end[..., 0]
        )
        - offset_rads
    )
