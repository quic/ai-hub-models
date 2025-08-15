# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from qai_hub_models.models._shared.sam.model_patches import Conv2DInplaceLinear
from qai_hub_models.utils.asset_loaders import SourceAsRoot

SAM2_SOURCE_REPO = "https://github.com/facebookresearch/sam2"
SAM2_SOURCE_REPO_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

with SourceAsRoot(
    SAM2_SOURCE_REPO,
    SAM2_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
):
    from sam2.modeling.backbones.hieradet import MLP as SAM2MaskDecoderMLP
    from sam2.modeling.backbones.hieradet import (
        MultiScaleBlock as SAMEncoderAttentionBlock,
    )
    from sam2.modeling.backbones.hieradet import do_pool
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
    from sam2.utils.transforms import SAM2Transforms  # noqa: F401


class Conv2DInplaceLinearSAMTransformerMLPBlock(nn.Module):
    """
    SAM MLPBlock that uses 1x1 Conv2D in place of linear layers.
    """

    def __init__(self, mlp_block: SAM2MaskDecoderMLP) -> None:
        super().__init__()
        self.lin1 = Conv2DInplaceLinear.from_linear(mlp_block.layers[0])
        self.lin2 = Conv2DInplaceLinear.from_linear(mlp_block.layers[1])
        self.act = mlp_block.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SplitHeadSAMEncoderAttention(nn.Module):
    """
    SAM Attention block with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
    """

    def __init__(self, attention_block: SAMEncoderAttentionBlock) -> None:
        super().__init__()
        self.out_feature, self.in_feature = (
            attention_block.qkv.weight.shape[0] // 3,
            attention_block.qkv.weight.shape[1],
        )

        bias = attention_block.qkv.bias[: self.out_feature] is not None
        self.q = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.k = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.v = Conv2DInplaceLinear(self.in_feature, self.out_feature, has_bias=bias)
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.num_heads = attention_block.num_heads
        self.q_pool = attention_block.q_pool

        for chunk, projList in enumerate([self.q, self.k, self.v]):
            projList.conv2d.weight.data.copy_(
                attention_block.qkv.weight[
                    (chunk) * self.out_feature : (chunk + 1) * self.out_feature,
                    :,
                    None,
                    None,
                ]
            )

            projList.conv2d.bias.data.copy_(
                attention_block.qkv.bias[
                    (chunk) * self.out_feature : (chunk + 1) * self.out_feature,
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        """
        #original code
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        """
        k = (
            self.k(x)
            .reshape(B, H * W, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, H * W, -1)
        )
        v = (
            self.v(x)
            .reshape(B, H * W, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, H * W, -1)
        )

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = self.q(x)
            q = do_pool(q, self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)
            q = q.permute(0, 2, 1, 3)
            q = q.reshape(B * self.num_heads, H * W, -1)
        else:
            q = (
                self.q(x)
                .reshape(B, H * W, self.num_heads, -1)
                .permute(0, 2, 1, 3)
                .reshape(B * self.num_heads, H * W, -1)
            )
        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )

        # Transpose back
        x = x.reshape(B, self.num_heads, H * W, -1)
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class SAM2Normalize(nn.Module):
    """
    Normalization module for SAM2, adapted from `sam2.utils.transforms.SAM2Transforms`.

    This version excludes the resizing operation, as resizing is handled externally
    and not within the model's forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = torch.jit.script(
            nn.Sequential(
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        return self.transforms(x)


def sam_decoder_predict_masks(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    repeat_image: bool,
    high_res_features: Optional[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predicts segmentation masks using the SAM decoder architecture with optional high-resolution features.

    This method is a modified version of `SamMaskDecoder.predict_masks`, designed to avoid unnecessary
    per-image batch expansion when not required. This helps prevent issues during QNN (Qualcomm Neural Network)
    compilation, such as the introduction of no-op operations or unsupported tensor dimensions.

    Args:
        image_embeddings (torch.Tensor): Image feature embeddings from the encoder.
        image_pe (torch.Tensor): Positional encodings for the image.
        sparse_prompt_embeddings (torch.Tensor): Sparse prompt tokens (e.g., points, boxes).
        dense_prompt_embeddings (torch.Tensor): Dense prompt features (e.g., masks).
        repeat_image (bool): Whether to repeat image embeddings for each prompt.
        high_res_features (Optional[list[torch.Tensor]]): Optional high-resolution features for refinement.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - masks: Predicted segmentation masks.
            - iou_pred: Predicted IoU scores for each mask.
            - mask_tokens_out: Output embeddings for each mask token.
            - object_score_logits: Object presence confidence scores.
    """
    # Concatenate output tokens
    s = 0
    if self.pred_obj_scores:
        output_tokens = torch.cat(
            [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            dim=0,
        )
        s = 1
    else:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0), -1, -1
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    # Expand per-image data in batch direction to be per-mask
    # -- Begin Qualcomm Modification --
    num_mask = tokens.shape[0]
    if num_mask != 1:
        tile_dims = [1] * len(image_embeddings.shape)
        tile_dims[0] = num_mask
        src = torch.tile(image_embeddings, tile_dims)

        tile_dims = [1] * len(image_pe.shape)
        tile_dims[0] = num_mask
        pos_src = torch.tile(image_pe, tile_dims)
    else:
        src = image_embeddings
        pos_src = image_pe

    src = src + dense_prompt_embeddings
    b, c, h, w = src.shape
    # -- End Qualcomm Modification --

    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, s, :]
    mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    if not self.use_high_res_features or high_res_features is None:
        upscaled_embedding = self.output_upscaling(src)
    else:
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

    hyper_in_list: list[torch.Tensor] = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        )
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)
    if self.pred_obj_scores:
        assert s == 1
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
    else:
        # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
        object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

    return masks, iou_pred, mask_tokens_out, object_score_logits


def mask_postprocessing(
    low_res_masks: torch.Tensor, orig_im_size: tuple[int, int]
) -> torch.Tensor:
    """
    Upscales low-resolution masks to match the original image size using bilinear interpolation.

    This function is adapted from `sam2.utils.transforms.SAM2Transforms.postprocess_masks`,
    modified to be used independently of the decoder class instance.

    Args:
        low_res_masks (torch.Tensor): A tensor of low-resolution masks, typically output from a model.
        orig_im_size (tuple[int, int]): The original image size as (height, width) to which the masks should be resized.

    Returns:
        torch.Tensor: A tensor of masks resized to the original image dimensions.
    """
    masks = torch.nn.functional.interpolate(
        low_res_masks,
        size=(orig_im_size[0], orig_im_size[1]),
        mode="bilinear",
        align_corners=False,
    )
    return masks
