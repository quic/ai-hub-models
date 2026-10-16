# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import cast

import torch
from mobile_sam import sam_model_registry
from mobile_sam.modeling.mask_decoder import MLP as SAMMaskDecoderMLP
from mobile_sam.modeling.sam import Sam
from mobile_sam.modeling.transformer import TwoWayAttentionBlock, TwoWayTransformer
from mobile_sam.utils.onnx import SamOnnxModel
from typing_extensions import Self

from qai_hub_models import TargetRuntime
from qai_hub_models.datasets.coco import CocoSamPointDataset
from qai_hub_models.models.templates.sam.evaluator import SAMMaskIoUEvaluator
from qai_hub_models.models.templates.sam.model_patches import (
    Conv2DInplaceLinearSAMMaskDecoderMLP,
    Conv2DInplaceLinearSAMTransformerMLPBlock,
    SplitHeadSAMDecoderAttention,
    sam_decoder_predict_masks,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
SMALL_MODEL_TYPE = "vit_t"
DEFAULT_MODEL_TYPE = SMALL_MODEL_TYPE
MODEL_REGISTRY = {
    SMALL_MODEL_TYPE: "mobile_sam.pt",
}
MODEL_ASSET_VERSION = 1
MODEL_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, MODEL_REGISTRY[SMALL_MODEL_TYPE]
)


class MobileSAMEncoder(BaseModel):
    def __init__(self, sam: Sam) -> None:
        super().__init__()
        self.sam = sam

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.sam.pixel_mean) * (1 / self.sam.pixel_std)
        return self.sam.image_encoder(x)

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(
                    batch_size,
                    3,
                    self.sam.image_encoder.img_size,
                    self.sam.image_encoder.img_size,
                ),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "image_embeddings": TensorSpec(apply_runtime_channel_reordering=True),
        }

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        return cls(MobileSAMLoader._load_sam_from_repo(model_type))


class MobileSAMDecoder(BaseModel):
    """
    Adapted from from segment_anything.utils.onnx.SamOnnxModel with modifications.

    This removes output mask resizing. Because this requires a dynamic shape to accomplish
    in the network, it's better to do this as a postprocessing step rather than in the inference
    framework itself.
    """

    def __init__(self, sam: Sam, return_single_mask: bool) -> None:
        super().__init__(sam)
        self.model: Sam
        self.embed_size = self.model.prompt_encoder.image_embedding_size
        self.img_size = sam.image_encoder.img_size
        self.return_single_mask = return_single_mask

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Lifted from segment_anything.utils.onnx.SamOnnxModel

        Modified to cast the boolean factors to the proper type first (see factor =)
        This allows torch.onnx.export to pass on this model
        """
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        factor = (point_labels != -1).to(point_embedding.dtype)
        point_embedding = point_embedding * factor
        factor = (point_labels == -1).to(point_embedding.dtype)
        point_embedding = (
            point_embedding
            + self.model.prompt_encoder.not_a_point_embed.weight * factor
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            factor = (point_labels == i).to(point_embedding.dtype)
            point_embedding = (
                point_embedding
                + self.model.prompt_encoder.point_embeddings[i].weight * factor
            )

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Lifted from segment_anything.utils.onnx.SamOnnxModel

        Modified to remove ops based on whether input_mask is set.
        """
        if input_mask is not None:
            return self.model.prompt_encoder.mask_downscaling(input_mask)
        return torch.zeros(
            1, 1, *self.embed_size
        ) + self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM lightweight decoder and return generated mask for given points

        Parameters
        ----------
        image_embeddings
            Image embeddings generated by Encoder. Shape [1, emb_dim, emb_size, emb_size]
        point_coords
            Point coordinates from input image for segmentation. Shape [1, k, 2]
        point_labels
            Point Labels to select/de-select given point for segmentation. Shape [1, k]
            e.g. Corresponding value is 1 if this point is to be included, otherwise 0
        mask_input
            Input mask to consider for segmentation. Shape [1, 1, 4 * self.embed_size, 4 * self.embed_size]
            If using point based segmentation, this is unused.

        Returns
        -------
        masks : torch.Tensor
            Generated masks. Shape [1, k, 256, 256]
        scores : torch.Tensor
            Mask scores. Shape [1, k]
            Note: k = number of points
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input)

        masks, scores = sam_decoder_predict_masks(
            self.model.mask_decoder,
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, scores = SamOnnxModel.select_masks(
                self, masks, scores, point_coords.shape[1]
            )

        return masks, scores

    def get_input_spec(
        self,
        has_mask_input: bool = False,
        num_of_points: int = 1,
    ) -> InputSpec:
        embed_dim = self.model.prompt_encoder.embed_dim
        embed_size = self.embed_size
        mask_input_size = tuple([4 * x for x in embed_size])

        input_spec: InputSpec = {
            "image_embeddings": TensorSpec(
                shape=(1, embed_dim, *embed_size),
                dtype="float32",
                io_type=IoType.TENSOR,
                apply_runtime_channel_reordering=True,
            ),
            "point_coords": TensorSpec(
                shape=(1, num_of_points, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "point_labels": TensorSpec(
                shape=(1, num_of_points),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }
        if has_mask_input:
            input_spec["mask_input"] = TensorSpec(
                shape=(1, 1, *mask_input_size),
                dtype="float32",
                io_type=IoType.TENSOR,
                apply_runtime_channel_reordering=True,
            )
            input_spec["has_mask_input"] = TensorSpec(
                shape=(1,),
                dtype="float32",
                io_type=IoType.TENSOR,
            )
        return input_spec

    def get_output_spec(self) -> OutputSpec:
        return {
            "masks": TensorSpec(apply_runtime_channel_reordering=True),
            "scores": TensorSpec(),
        }

    @classmethod
    def from_pretrained(
        cls, model_type: str = DEFAULT_MODEL_TYPE, single_mask_mode: bool = True
    ) -> Self:
        return cls(
            MobileSAMLoader._load_sam_from_repo(model_type),
            return_single_mask=single_mask_mode,
        )


class MobileSAMLoader:
    @classmethod
    def load(
        cls,
        model_type: str = DEFAULT_MODEL_TYPE,
        single_mask_mode: bool = True,
    ) -> tuple[Sam, MobileSAMEncoder, MobileSAMDecoder]:
        sam = cls._load_sam_from_repo(model_type)
        cls._patch_mobilesam_for_qnn_comatibility(sam)
        return sam, MobileSAMEncoder(sam), MobileSAMDecoder(sam, single_mask_mode)

    @staticmethod
    def _load_sam_from_repo(model_type: str = DEFAULT_MODEL_TYPE) -> Sam:
        weight_asset = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, MODEL_REGISTRY[model_type]
        )
        weight_asset.fetch()
        return sam_model_registry[model_type](weight_asset.path)

    @staticmethod
    def _patch_mobilesam_for_qnn_comatibility(sam: Sam) -> None:
        """Apply a patch to the SAM class for compatibility with QNN."""
        # Scale mean/std from [0-255] space to [0-1] space so the
        # normalization in MobileSAMEncoder.forward works with float [0, 1] inputs.
        sam.pixel_mean = sam.pixel_mean / 255.0
        sam.pixel_std = sam.pixel_std / 255.0

        ###
        # Patch the graph for compatibility with QNN.
        #
        # All below optimizations either optimize for QNN inference speed,
        # or fix failures that occur when compiling to QNN.
        ###
        sam.mask_decoder.predict_masks = functools.partial(
            sam_decoder_predict_masks, sam.mask_decoder
        )
        for i in range(len(sam.mask_decoder.output_hypernetworks_mlps)):
            mlp = cast(SAMMaskDecoderMLP, sam.mask_decoder.output_hypernetworks_mlps[i])
            sam.mask_decoder.output_hypernetworks_mlps[i] = (
                Conv2DInplaceLinearSAMMaskDecoderMLP(mlp)
            )
        sam.mask_decoder.iou_prediction_head = Conv2DInplaceLinearSAMMaskDecoderMLP(
            sam.mask_decoder.iou_prediction_head
        )

        transformer = cast(TwoWayTransformer, sam.mask_decoder.transformer)
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


class MobileSAM(WorkbenchModelCollection):
    def __init__(
        self, sam: Sam, encoder: MobileSAMEncoder, decoder: MobileSAMDecoder
    ) -> None:
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.sam = sam
        self.encoder = encoder
        self.decoder = decoder

    def get_input_spec(
        self,
        batch_size: int = 1,
        has_mask_input: bool = False,
        num_of_points: int = 1,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            has_mask_input=has_mask_input,
            num_of_points=num_of_points,
        )

    @classmethod
    def from_pretrained(
        cls, model_type: str = DEFAULT_MODEL_TYPE, single_mask_mode: bool = True
    ) -> Self:
        return cls(*MobileSAMLoader.load(model_type, single_mask_mode))

    def get_component_hub_link_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        # TODO: O=1 to work-around a tiling bug (AISW-206993)
        return super().get_component_hub_link_options(
            component_name,
            target_runtime,
            f"{other_link_options} --qnn_options default_graph_htp_optimizations=O=1",
        )

    def get_evaluator(self) -> BaseEvaluator:
        return SAMMaskIoUEvaluator(self.sam.mask_threshold)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoSamPointDataset]
