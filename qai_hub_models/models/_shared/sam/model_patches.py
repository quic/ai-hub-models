# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from math import floor
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def sam_decoder_predict_masks(
    self,  # SAMMaskDecoder
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
):
    """
    SamMaskDecoder.predict_masks modified to skip the per-image batch expansion if no expansion is required.

    If no expansion is required, a noop is left in the graph, which causes compilation to QNN to fail.

    Repeat-interleave also generates a 5D tensor, which causes compilation to fail, so it is replaced with Tile.
    """
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
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
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = self.output_upscaling(src)
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

    return masks, iou_pred


class Conv2DInplaceLinear(nn.Module):
    """
    An implementation of Linear / Conv1D that uses a 1x1 Conv2D op instead.

    The Conv2D implementation for Qualcomm DSPs is faster than the Linear/Conv1D implementation.
    """

    @staticmethod
    def from_linear(mod: torch.nn.Linear | torch.nn.Conv1d) -> Conv2DInplaceLinear:
        if isinstance(mod, torch.nn.Linear):
            weight, bias = mod.weight, mod.bias
            bias = mod.bias
        elif isinstance(mod, torch.nn.Conv1d):
            weight, bias = mod.weight.T, mod.bias
        else:
            raise NotImplementedError()

        out_features, in_features = weight.shape
        linear = Conv2DInplaceLinear(
            in_features,
            out_features,
            bias is not None,
            mod.device if hasattr(mod, "device") else None,
        )
        linear.conv2d.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            assert linear.conv2d.bias is not None
            linear.conv2d.bias.data.copy_(bias.data)

        return linear

    def __init__(
        self,
        in_features,
        out_features,
        has_bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_features, out_features, 1, bias=has_bias)
        if device:
            self.conv2d.to(device)

    def __getattr__(self, attr):
        conv2d = self._modules["conv2d"]
        if attr == "conv2d":
            return conv2d
        return getattr(conv2d, attr)

    def forward(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        elif ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            pass

        x = x.permute(0, 3, 1, 2)  # (B, L, D) -> (B, D, 1, L)
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)

        if ndim == 2:
            x = x.squeeze(1).squeeze(0)
        elif ndim == 3:
            x = x.squeeze(1)
        elif ndim == 4:
            pass
        return x


class Conv2DInplaceLinearSAMMaskDecoderMLP(nn.Module):
    """
    SAM MLP that uses 1x1 Conv2D in place of linear layers.
    """

    def __init__(self, mlp):  # from segment_anything.modeling.mask_decoder import MLP
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = mlp.num_layers
        self.sigmoid_output = mlp.sigmoid_output
        for module in mlp.layers:
            assert isinstance(module, nn.Linear)
            self.layers.append(Conv2DInplaceLinear.from_linear(module))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Conv2DInplaceLinearSAMTransformerMLPBlock(nn.Module):
    """
    SAM MLPBlock that uses 1x1 Conv2D in place of linear layers.
    """

    def __init__(
        self, mlp_block  # from segment_anything.modeling.image_encoder import MLPBlock,
    ):
        super().__init__()
        self.lin1 = Conv2DInplaceLinear.from_linear(mlp_block.lin1)
        self.lin2 = Conv2DInplaceLinear.from_linear(mlp_block.lin2)
        self.act = mlp_block.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SplitHeadSAMDecoderAttention(nn.Module):
    def __init__(
        self,
        attention_block,  # from segment_anything.modeling.transformer import Attention
    ):
        super().__init__()
        self.embedding_dim = attention_block.embedding_dim  # in features
        self.internal_dim = attention_block.internal_dim  # out features

        self.num_heads = attention_block.num_heads
        self.in_features = self.embedding_dim
        self.out_features = self.internal_dim
        self.qkv_out_features_per_head = self.out_features // self.num_heads

        self.qproj = torch.nn.ModuleList()
        self.kproj = torch.nn.ModuleList()
        self.vproj = torch.nn.ModuleList()
        self.out_proj = attention_block.out_proj
        for i in range(attention_block.num_heads):
            for projList, merged_layer in [
                (self.qproj, attention_block.q_proj),
                (self.kproj, attention_block.k_proj),
                (self.vproj, attention_block.v_proj),
            ]:
                split_layer = Conv2DInplaceLinear(
                    self.in_features, self.qkv_out_features_per_head
                )
                split_layer.conv2d.weight.data.copy_(
                    merged_layer.weight[
                        i
                        * self.qkv_out_features_per_head : (i + 1)
                        * self.qkv_out_features_per_head,
                        :,
                        None,
                        None,
                    ]
                )

                split_layer.conv2d.bias.data.copy_(
                    merged_layer.bias[
                        i
                        * self.qkv_out_features_per_head : (i + 1)
                        * self.qkv_out_features_per_head
                    ]
                )
                projList.append(split_layer)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        attns = []
        for i in range(0, self.num_heads):
            # Single head attention
            qOut: torch.Tensor = self.qproj[i](q)
            kOut: torch.Tensor = self.kproj[i](k)
            vOut: torch.Tensor = self.vproj[i](v)
            attn = qOut @ kOut.transpose(-2, -1)
            attn = attn / math.sqrt(self.qkv_out_features_per_head)
            attn = torch.softmax(attn, dim=-1)
            attns.append(attn @ vOut)

        # Combine heads
        return self.out_proj(torch.cat(attns, -1))


def resize_longest_image_size(
    input_image_size: tuple[int, int], longest_side: int
) -> tuple[int, int]:
    """
    Lifted from segment_anything.utils.onnx.SamOnnxModel.mask_postprocessing

    Modified to break this apart from the decoder class instance.
    """
    scale = longest_side / max(input_image_size)
    transformed_size = cast(
        tuple[int, int],
        tuple(int(floor(scale * each + 0.5)) for each in input_image_size),
    )
    return transformed_size


def mask_postprocessing(
    masks: torch.Tensor, encoder_img_size: int, orig_im_size: tuple[int, int]
) -> torch.Tensor:
    """
    Lifted from segment_anything.utils.onnx.SamOnnxModel.mask_postprocessing

    Modified to break this apart from the decoder class instance.
    """
    masks = F.interpolate(
        masks,
        size=(encoder_img_size, encoder_img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, encoder_img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks
