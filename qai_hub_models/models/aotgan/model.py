# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.inpaint_evaluator import InpaintEvaluator
from qai_hub_models.models._shared.repaint.model import RepaintModel
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    wipe_sys_modules,
)

AOTGAN_SOURCE_REPOSITORY = "https://github.com/researchmm/AOT-GAN-for-Inpainting/"
AOTGAN_SOURCE_REPO_COMMIT = "418034627392289bdfc118d62bc49e6abd3bb185"
AOTGAN_SOURCE_PATCHES = [
    # Prevent overflow in layer norm (and re-use mean)
    # On both on TFLite/QNN, the divider by (n - 1) ends up before the sum, so
    # overflow is avoided.
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "layer_norm.diff")
    )
]
MODEL_ID = __name__.split(".")[-2]
SUPPORTED_PRETRAINED_MODELS = {"celebahq", "places2"}
DEFAULT_WEIGHTS = "celebahq"
MODEL_ASSET_VERSION = 3


class AOTGAN(RepaintModel):
    """Exportable AOTGAN for Image inpainting"""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        if ckpt_name not in SUPPORTED_PRETRAINED_MODELS:
            raise ValueError(
                "Unsupported pre_trained model requested. Please provide either 'celeabhq' or 'places2'."
            )
        downloaded_model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"pretrained_models/{ckpt_name}/G0000000.pt",
        ).fetch()
        with SourceAsRoot(
            AOTGAN_SOURCE_REPOSITORY,
            AOTGAN_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=AOTGAN_SOURCE_PATCHES,
        ):
            import src

            wipe_sys_modules(src)

            from src.model.aotgan import InpaintGenerator

            # AOT-GAN InpaintGenerator uses ArgParser to
            # initialize model and it uses following two parameters
            #  - rates: default value [1, 2, 4, 8]
            #  - block_num: default value 8
            # creating dummy class with default values to set the same
            class InpaintArgs:
                def __init__(self):
                    self.rates = [1, 2, 4, 8]
                    self.block_num = 8

            args = InpaintArgs()
            model = InpaintGenerator(args)
            model.load_state_dict(torch.load(downloaded_model_path, map_location="cpu"))
            return cls(model)

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Run AOTGAN Inpaint Generator on `image` with given `mask`
        and generates new high-resolution in-painted image.

        Parameters:
            image: Image to which the mask should be applied. [N, C, H, W]
                    Range: float[0, 1]
                    3-channel color Space: RGB
            mask: Pixel values pre-processed to have have mask values either 0. or 1.
                    Range: float[0, 1] and only values of 0. or 1.
                    1-channel binary image.

        Returns:
            In-painted image for given image and mask of shape [N, C, H, W]
            Range: float[0, 1]
            3-channel color space: RGB
        """
        image_normalized_rgb = image * 2 - 1
        image_normalized_bgr = torch.flip(image_normalized_rgb, dims=[1])
        image_masked_normalized_bgr = (image_normalized_bgr * (1 - mask).float()) + mask

        pred_bgr = self.model(image_masked_normalized_bgr, mask)
        # Uncomment to overlay the mask over the prediction. Useful for debugging.
        # pred_bgr = pred_bgr * mask + image * (1 - mask)

        pred_rgb = torch.flip(pred_bgr, dims=[1])
        pred_rgb_clamped = torch.clamp(pred_rgb, -1, 1)
        pred_rgb_unnormalized = (pred_rgb_clamped + 1) / 2
        return pred_rgb_unnormalized

    def get_evaluator(self) -> BaseEvaluator:
        return InpaintEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["celebahq"]
