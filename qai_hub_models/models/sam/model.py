# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

# This is a fork of https://github.com/facebookresearch/segment-anything
# with changes to make the SAM decoder traceable
SAM_SOURCE_REPO = "https://github.com/dmckinnon/segment-anything"
SAM_SOURCE_REPO_COMMIT = "0bc06e062ca883c2524bfa79061807c535eb0d51"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_MODEL_TYPE = "vit_l"
SMALL_MODEL_TYPE = "vit_b"
MODEL_REGISTERY = {
    "vit_b": "sam_vit_b_01ec64.pth",  # 91M params
    "vit_l": "sam_vit_l_0b3195.pth",  # 308M params
    "vit_h": "sam_vit_h_4b8939.pth",  # 636M params
}
MODEL_ASSET_VERSION = 1


class SAMQAIHMWrapper(CollectionModel):
    """
    QAIHM version of segment-anything (https://github.com/dmckinnon/segment-anything)

    QAIHM fork modifies following from parent segment-anything repo:
        1. window_partition in encoder works on rank-5 tensor instead of rank-6 tensor
        2. SamOnnxModel accepts `orig_img_size` to use static upsample instead of dynamic upsample
    """

    def __init__(
        self,
        sam: torch.nn.Module,
        sam_encoder: Callable,
        SamOnnxModel,
        ResizeLongestSide,
        SamPredictor,
    ):
        self.sam = sam
        self.sam_encoder = sam_encoder
        self.SamOnnxModel = SamOnnxModel
        self.ResizeLongestSide = ResizeLongestSide
        self.SamPredictor = SamPredictor

    def get_sam(self) -> torch.nn.Module:
        return self.sam

    def get_sam_encoder(self) -> Callable:
        return self.sam_encoder

    # Create a new decoder
    def get_sam_decoder(
        self, orig_img_size: Tuple[int, int] = (720, 1280), single_mask_mode=True
    ) -> Callable:
        self.sam_decoder = SegmentAnythingONNXDecoder(
            self,
            single_mask_mode=single_mask_mode,
            orig_img_size=orig_img_size,
        )
        return self.sam_decoder

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> SAMQAIHMWrapper:
        with SourceAsRoot(
            SAM_SOURCE_REPO, SAM_SOURCE_REPO_COMMIT, MODEL_ID, MODEL_ASSET_VERSION
        ):
            from segment_anything import SamPredictor, sam_model_registry
            from segment_anything.utils.onnx import SamOnnxModel
            from segment_anything.utils.transforms import ResizeLongestSide

            sam = sam_model_registry[model_type](_get_weights_path(model_type))
            sam_encoder = SegmentAnythingEncoder(sam, ResizeLongestSide)
            return cls(sam, sam_encoder, SamOnnxModel, ResizeLongestSide, SamPredictor)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot call SAMQAIHMWrapper directly")


class SegmentAnythingEncoder(BaseModel):
    """Exportable SAM encoder"""

    def __init__(
        self,
        sam: torch.nn.Module,
        ResizeLongestSide: Callable,
    ) -> None:
        super().__init__()
        self.sam = sam
        self.transforms = ResizeLongestSide(self.sam.image_encoder.img_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run SAM Image encoder and returns image embeddings

        Parameters:
            image: Pixel values pre-procewindow_partitionssed for encoder consumption.
                   Range: float[0, 255] normalized via preprocess_input_image
                   3-channel Color Space: RGB

        Returns:
            image_embeddings
        """
        return self.sam.image_encoder(image)

    def _get_input_spec_for_instance(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(
            batch_size,
            self.sam.image_encoder.img_size,
            self.sam.image_encoder.img_size,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        encoder_img_height: int = 1024,  # self.sam.image_encoder.img_size[0]
        encoder_img_width: int = 1024,  # self.sam.image_encoder.img_size[1]
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": (
                (batch_size, 3, encoder_img_height, encoder_img_width),
                "float32",
            )
        }

    @staticmethod
    def get_output_names() -> List[str]:
        return ["image_embeddings"]

    def preprocess_input_image(self, input_image: np.ndarray):
        """Transform input image to work with SAM encoder"""
        transformed_image = torch.as_tensor(
            self.transforms.apply_image(input_image)
        ).type(torch.float32)
        transformed_image = transformed_image.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]

        self.input_size = transformed_image.shape[-2:]
        self.original_size = input_image.shape[:2]
        return self.sam.preprocess(transformed_image)

    @classmethod
    def from_pretrained(cls):
        return SAMQAIHMWrapper.from_pretrained().get_sam_encoder()


class SegmentAnythingONNXDecoder(BaseModel):
    """Exportable SAM decoder"""

    def __init__(
        self,
        sam_qaihm_wrapper: SAMQAIHMWrapper,
        orig_img_size: Tuple[int, int] = (720, 1280),
        single_mask_mode: bool = True,
    ) -> None:
        super().__init__()
        self.sam = sam_qaihm_wrapper.get_sam()
        self.sam_decoder = sam_qaihm_wrapper.SamOnnxModel(
            self.sam, orig_img_size=orig_img_size, return_single_mask=single_mask_mode
        )
        self.transforms = sam_qaihm_wrapper.ResizeLongestSide(
            self.sam.image_encoder.img_size
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run SAM lightweight decoder and return generated mask for given points

        Parameters:
            image_embeddings: torch.Tensor of shape [1, emb_dim, emb_size, emb_size]
                Image embeddings generated by Encoder
            point_coords: torch.Tensor of shape [1, k, 2]
                Point coordinates from input image for segmentation
            point_labels: torch.Tensor of shape [1, k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0
            mask_input: torch.Tensor of shape [1, 1, 4 * image_emd_size, 4 * image_emb_size]
                Input mask to consider for segmentation. If using point based segmentation, set this to torch.zeros()
            has_mask_input: torch.Tensor of shape [1]
                If has value [1] then mask_input is used, otherwise no.
                If using point based segmentation, can set this to [0]

        Returns:
            upscaled_masks: torch.Tensor of shape [1, k, <input image spatial dims>]
            score: torch.Tensor of shape [1, k]
            masks: torch.Tensor of shape [1, k, 256, 256]
                Use this low resolution masks to further slice and upscale for resolutions that Decoder is not intialized to.

        Where,
            k = number of points
        """
        return self.sam_decoder(
            image_embeddings, point_coords, point_labels, mask_input, has_mask_input
        )

    def _get_input_spec_for_instance(
        self,
        num_of_points: int = 1,
    ) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(
            num_of_points,
            self.sam.prompt_encoder.embed_dim,
            self.sam.prompt_encoder.image_embedding_size[0],
            self.sam.prompt_encoder.image_embedding_size[1],
        )

    @staticmethod
    def get_input_spec(
        num_of_points: int = 1,
        embed_dim: int = 256,
        image_embedding_height: int = 64,
        image_embedding_width: int = 64,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        embed_size = (image_embedding_height, image_embedding_width)
        mask_input_size = [4 * x for x in embed_size]

        input_spec = {
            "image_embeddings": ((1, embed_dim, *embed_size), "float32"),
            "point_coords": ((1, num_of_points, 2), "float32"),
            "point_labels": ((1, num_of_points), "float32"),
            "mask_input": ((1, 1, *mask_input_size), "float32"),
            "has_mask_input": ((1,), "float32"),
        }
        return input_spec

    @staticmethod
    def get_output_names() -> List[str]:
        return ["upscaled_masks", "scores", "masks"]

    @classmethod
    def from_pretrained(cls):
        return SAMQAIHMWrapper.from_pretrained().get_sam_decoder()


def _get_weights_path(model_type: str = DEFAULT_MODEL_TYPE) -> Path:
    """Convert from names of weights files to the url for the weights file"""
    if model_type not in MODEL_REGISTERY.keys():
        raise RuntimeError(f"Weights not found for model type `{model_type}`.")

    asset = CachedWebModelAsset(
        f"https://dl.fbaipublicfiles.com/segment_anything/{MODEL_REGISTERY[model_type]}",
        MODEL_ID,
        MODEL_ASSET_VERSION,
        f"{MODEL_REGISTERY[model_type]}",
    )
    asset.fetch()
    return asset.path()
