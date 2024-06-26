# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image as ImageModule
from PIL.Image import Image

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CITYSCAPES_MEAN,
    CITYSCAPES_STD,
    FFNET_SOURCE_PATCHES,
    FFNET_SOURCE_REPO_COMMIT,
    FFNET_SOURCE_REPOSITORY,
    FFNET_SOURCE_VERSION,
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, SourceAsRoot
from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad


def _load_cityscapes_loader(cityscapes_path: Optional[str] = None) -> object:
    if cityscapes_path is None:
        # Allow a loader without data. There are useful auxiliary functions.
        cityscapes_path = ASSET_CONFIG.get_local_store_model_path(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            "cityscapes_dummy",
        )

        os.makedirs(
            os.path.join(cityscapes_path, "leftImg8bit", "train"), exist_ok=True
        )
        os.makedirs(os.path.join(cityscapes_path, "leftImg8bit", "val"), exist_ok=True)

    # Resolve absolute path outside SourceAsRoot, since cwd changes
    cityscapes_path = os.path.abspath(cityscapes_path)

    with SourceAsRoot(
        FFNET_SOURCE_REPOSITORY,
        FFNET_SOURCE_REPO_COMMIT,
        MODEL_ID,
        FFNET_SOURCE_VERSION,
        source_repo_patches=FFNET_SOURCE_PATCHES,
    ):
        import config

        config.cityscapes_base_path = cityscapes_path
        from ffnet_datasets.cityscapes.dataloader.get_dataloaders import (
            return_dataloader,
        )

        dataloader = return_dataloader(num_workers=1, batch_size=1)
        return dataloader


def preprocess_cityscapes_image(image: Image) -> torch.Tensor:
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ]
    )
    out_tensor: torch.Tensor = transform(image)  # type: ignore
    return out_tensor.unsqueeze(0)


class CityscapesSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform
    end to end inference for single-view (left) semantic segmentation of the
    Cityscapes (https://cityscapes-dataset.com/) dataset.

    The app uses 1 model:
        * Cityscapes segmentation model

    For a given image input, the app will:
        * Pre-process the image
        * Run model inference
        * Resize predictions to map image size
        * Visualize results by super-imposing on input image
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_specs: Mapping[str, Tuple[Tuple[int, ...], str]],
    ):
        self.model = model
        self.color_mapping = _load_cityscapes_loader().dataset.color_mapping
        (_, _, self.model_height, self.model_width) = input_specs["image"][0]

    def predict(self, image: Image, raw_output: bool = False) -> Image | np.ndarray:
        """
        From the provided image or tensor, predict semantic segmentation over
        the Cityscapes classes.

        Parameters:
            image: A PIL Image in RGB format.

        Returns:
            If raw_output is False it will return an annotated image of the
            same size as the input image. If True, it will return raw logit
            probabilities as an numpy array of shape [1, CLASSES, HEIGHT,
            WIDTH]. Note, that WIDTH and HEIGHT will be smaller than the input
            image.
        """
        resized_image, scale, padding = pil_resize_pad(
            image, (self.model_height, self.model_width)
        )

        input_tensor = preprocess_cityscapes_image(resized_image)
        small_res_output = self.model(input_tensor)

        output = F.interpolate(
            small_res_output,
            (resized_image.height, resized_image.width),
            mode="bilinear",
        )
        if raw_output:
            return output.detach().numpy()
        predictions = output[0].argmax(0).byte().cpu().numpy()

        color_mask = ImageModule.fromarray(predictions.astype(np.uint8)).convert("P")
        color_mask.putpalette(self.color_mapping)
        out = ImageModule.blend(resized_image, color_mask.convert("RGB"), 0.5)

        # Resize / unpad annotated image
        image_annotated = pil_undo_resize_pad(out, image.size, scale, padding)

        return image_annotated
