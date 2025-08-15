# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import os
from typing import Optional, cast

import torch
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from qai_hub.client import Device

from qai_hub_models.models._shared.sam.model_patches import (
    Conv2DInplaceLinearSAMMaskDecoderMLP,
    SplitHeadSAMDecoderAttention,
)
from qai_hub_models.models.sam2.model_patches import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM2_SOURCE_REPO,
    SAM2_SOURCE_REPO_COMMIT,
    Conv2DInplaceLinearSAMTransformerMLPBlock,
    SAM2Normalize,
    SplitHeadSAMEncoderAttention,
    sam_decoder_predict_masks,
)
from qai_hub_models.models.sam2.utils import copy_configs
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

BASE_PLUS_MODEL_TYPE = "base_plus"
LARGE_MODEL_TYPE = "large"
SMALL_MODEL_TYPE = "small"
TINY_MODEL_TYPE = "tiny"
DEFAULT_MODEL_TYPE = TINY_MODEL_TYPE

MODEL_REGISTERY = {
    BASE_PLUS_MODEL_TYPE: "sam2.1_hiera_base_plus.pt",
    LARGE_MODEL_TYPE: "sam2.1_hiera_large.pt",
    SMALL_MODEL_TYPE: "sam2.1_hiera_small.pt",
    TINY_MODEL_TYPE: "sam2.1_hiera_tiny.pt",
}

CONFIG_REGISTERY = {
    TINY_MODEL_TYPE: "sam2.1_hiera_t",
    SMALL_MODEL_TYPE: "sam2.1_hiera_s",
    LARGE_MODEL_TYPE: "sam2.1_hiera_l",
    BASE_PLUS_MODEL_TYPE: "sam2.1_hiera_b+",
}

BB_FEAT_SIZES = [
    (256, 256),
    (128, 128),
    (64, 64),
]

with SourceAsRoot(
    SAM2_SOURCE_REPO,
    SAM2_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
) as repo_path:
    from sam2.build_sam import build_sam2
    from sam2.modeling.backbones.hieradet import MultiScaleBlock as SAM2_Encoder_Block
    from sam2.modeling.sam2_base import SAM2Base as Sam2
    from sam2.modeling.sam2_utils import MLP as SAM2MaskDecoderMLP
    from sam2.modeling.sam.transformer import TwoWayAttentionBlock, TwoWayTransformer


class SAM2Encoder(BaseModel):
    """Exportable SAM2 encoder that can be split into several parts."""

    def __init__(
        self,
        sam2: Sam2,
    ) -> None:
        super().__init__()
        self.sam2 = sam2
        self.normalize = SAM2Normalize()
        self._bb_feat_sizes = BB_FEAT_SIZES

    def forward(
        self, Image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run SAM2 Image encoder and returns image_embeddings, high_res_features1, high_res_features2

        Parameters:
            Image:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]

        Returns:
                image_embeddings: Shape (1, 256, 64, 64)
                high_res_features1: Shape (1, 32, 256, 256)
                high_res_features2: Shape (1, 64, 128, 128)
        """
        x = self.normalize(Image)
        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        return image_embeddings, high_res_features1, high_res_features2

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        encoder_img_height: int = 1024,  # self.sam2.image_size
        encoder_img_width: int = 1024,  # self.sam2.image_size
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

    def _get_input_spec_for_instance(self, batch_size: int = 1) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.
        """
        return self.__class__.get_input_spec(
            batch_size, self.sam2.image_size, self.sam2.image_size
        )

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return list(SAM2Encoder.get_input_spec().keys())

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["image_embeddings", "high_res_features1", "high_res_features2"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["image_embeddings", "high_res_features1", "high_res_features2"]

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> SAM2Encoder:
        return SAM2Loader.load(model_type)[1]


class SAM2Decoder(BaseModel):
    """.
    This SAM2Decoder is taken from the class SAM2ImagePredictor.predict from sam2.

    This removes output mask resizing. Because this requires a dynamic shape to accomplish
    in the network, it's better to do this as a postprocessing step rather than in the inference
    framework itself.
    """

    def __init__(self, sam2: Sam2) -> None:
        super().__init__()
        self.model = sam2
        self.mask_decoder = self.model.sam_mask_decoder
        self.prompt_encoder = self.model.sam_prompt_encoder
        self.prompt_encoder_embed_dim = self.model.sam_prompt_embed_dim
        self.embed_size = self.model.sam_prompt_encoder.image_embedding_size
        self._bb_feat_sizes = BB_FEAT_SIZES
        self.high_res_features1_dim = 32
        self.high_res_features2_dim = 64

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor,  # [1, 32, 256, 256]
        high_res_features2: torch.Tensor,  # [1, 64, 128, 128]
        unnorm_coords: torch.Tensor,  # [num_labels,num_points,2]
        labels: torch.Tensor,  # [num_labels,num_points]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM2 lightweight decoder and return generated mask for given points

        Parameters:
            image_embeddings: torch.Tensor of shape [1, emb_dim, emb_size, emb_size]
                Image embeddings generated by Encoder
            high_res_features1: torch.Tensor of shape [1, high_res_1_dim, high_res_1_size, high_res_1_size]
                First set of high-resolution features.
            high_res_features2: torch.Tensor of shape [1, high_res_2_dim, high_res_2_size, high_res_2_size]
                Second set of high-resolution features.
            unnorm_coords: torch.Tensor of shape [1, k, 2]
                Point coordinates from input image for segmentation, mapped to the resized image
            labels: torch.Tensor of shape [1, k]
                Point Labels to select/de-select given point for segmentation
                e.g. Corresponding value is 1 if this point is to be included, otherwise 0

        Returns:
            masks: torch.Tensor of shape [1, 1, 256, 256]
            scores: torch.Tensor of shape [1, 1]
        """
        sparse_embedding, dense_embedding = self.prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        return low_res_masks, iou_predictions

    def _get_input_spec_for_instance(
        self: SAM2Decoder,
        num_of_points: int = 2,
    ) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(
            num_of_points,
            self.prompt_encoder_embed_dim,
            self._bb_feat_sizes[2],
            self._bb_feat_sizes[1],
            self._bb_feat_sizes[0],
            self.high_res_features1_dim,
            self.high_res_features2_dim,
        )

    @staticmethod
    def get_input_spec(
        num_of_points: int = 2,
        embed_dim: int = 256,
        image_embedding: tuple = (64, 64),
        high_res_featutes2: tuple = (128, 128),
        high_res_featutes1: tuple = (256, 256),
        high_res_features1_dim: int = 32,
        high_res_features2_dim: int = 64,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.

        input_spec: InputSpec = {
            "image_embeddings": (tuple((1, embed_dim, *image_embedding)), "float32"),
            "high_res_features1": (
                tuple((1, high_res_features1_dim, *high_res_featutes1)),
                "float32",
            ),
            "high_res_features2": (
                tuple((1, high_res_features2_dim, *high_res_featutes2)),
                "float32",
            ),
            "unnorm_coords": (tuple((1, num_of_points, 2)), "float32"),
            "labels": (tuple((1, num_of_points)), "int32"),
        }
        return input_spec

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        out = ["image_embeddings", "high_res_features1", "high_res_features2"]
        return out

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["masks"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["masks", "scores"]

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> SAM2Decoder:
        return SAM2Loader.load(model_type)[2]


class SAM2Loader:
    """
    Helper class for loading and preparing a HTP-compatible SAM2 model.
    """

    @staticmethod
    def load(
        model_type: str = SMALL_MODEL_TYPE,
    ) -> tuple[Sam2, SAM2Encoder, SAM2Decoder]:

        sam2 = SAM2Loader._load_sam2_from_repo(model_type)
        SAM2Loader._patch_sam2_for_qnn_comatibility(sam2)
        encoder = SAM2Encoder(sam2)
        decoder = SAM2Decoder(sam2)

        return sam2, encoder, decoder

    @staticmethod
    def _load_sam2_from_repo(model_type: str = DEFAULT_MODEL_TYPE) -> Sam2:
        """
        Get the SAM2 described by the given model type.
        SAM2 will be patched for QNN compatibility.
        """
        model_cfg_path = "build/configs/sam2.1"
        GlobalHydra.instance().clear()
        initialize(
            config_path=str(model_cfg_path),
            job_name="sam2_inference",
            version_base=None,
        )
        config_dir = QAIHM_MODELS_ROOT / MODEL_ID / "build"
        os.makedirs(config_dir, exist_ok=True)
        copy_configs(os.path.join(repo_path, "sam2", "configs", "sam2.1"), config_dir)
        if model_type not in MODEL_REGISTERY.keys():
            raise RuntimeError(f"Weights not found for model type `{model_type}`.")

        asset = CachedWebModelAsset(
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{MODEL_REGISTERY[model_type]}",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{MODEL_REGISTERY[model_type]}",
        )
        asset.fetch()
        return build_sam2(
            CONFIG_REGISTERY[model_type], asset.local_cache_path, device="cpu"
        )

    @staticmethod
    def _patch_sam2_for_qnn_comatibility(sam2: Sam2) -> None:
        """Apply a patch to the SAM2 Encoder class for compatibility with QNN."""

        ###
        # Patch the graph for compatibility with QNN.
        #
        # All below optimizations either optimize for QNN inference speed,
        # or fix failures that occur when compiling to QNN.
        ###
        for block in sam2.image_encoder.trunk.blocks:
            assert isinstance(block, SAM2_Encoder_Block)
            block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)
            block.attn = SplitHeadSAMEncoderAttention(block.attn)

        sam2.sam_mask_decoder.predict_masks = functools.partial(
            sam_decoder_predict_masks, sam2.sam_mask_decoder
        )
        for i in range(0, len(sam2.sam_mask_decoder.output_hypernetworks_mlps)):
            mlp = cast(
                SAM2MaskDecoderMLP, sam2.sam_mask_decoder.output_hypernetworks_mlps[i]
            )
            sam2.sam_mask_decoder.output_hypernetworks_mlps[i] = (
                Conv2DInplaceLinearSAMMaskDecoderMLP(mlp)
            )

        sam2.sam_mask_decoder.iou_prediction_head = (
            Conv2DInplaceLinearSAMMaskDecoderMLP(
                sam2.sam_mask_decoder.iou_prediction_head
            )
        )

        transformer = cast(TwoWayTransformer, sam2.sam_mask_decoder.transformer)
        transformer.final_attn_token_to_image = SplitHeadSAMDecoderAttention(
            transformer.final_attn_token_to_image
        )
        for block in transformer.layers:
            block = cast(TwoWayAttentionBlock, block)
            block.self_attn = SplitHeadSAMDecoderAttention(block.self_attn)
            block.cross_attn_token_to_image = SplitHeadSAMDecoderAttention(
                block.cross_attn_token_to_image
            )
            block.cross_attn_image_to_token = SplitHeadSAMDecoderAttention(
                block.cross_attn_image_to_token
            )
            block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)


@CollectionModel.add_component(SAM2Encoder)
@CollectionModel.add_component(SAM2Decoder)
class SAM2(CollectionModel):
    def __init__(self, sam2: Sam2, encoder: SAM2Encoder, decoder: SAM2Decoder) -> None:
        super().__init__(*[encoder, decoder])
        self.sam2 = sam2
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> SAM2:
        return cls(*SAM2Loader.load(model_type))


class SAM2Tiny(SAM2):
    @classmethod
    def from_pretrained(cls) -> SAM2:
        return cls(*SAM2Loader.load(TINY_MODEL_TYPE))


class SAM2Small(SAM2):
    @classmethod
    def from_pretrained(cls) -> SAM2:
        return cls(*SAM2Loader.load(SMALL_MODEL_TYPE))


class SAM2BasePlus(SAM2):
    @classmethod
    def from_pretrained(cls) -> SAM2:
        return cls(*SAM2Loader.load(BASE_PLUS_MODEL_TYPE))


class SAM2Large(SAM2):
    @classmethod
    def from_pretrained(cls) -> SAM2:
        return cls(*SAM2Loader.load(LARGE_MODEL_TYPE))
