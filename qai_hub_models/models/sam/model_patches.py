# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from math import floor
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from qai_hub_models.utils.asset_loaders import SourceAsRoot

SAM_SOURCE_REPO = "https://github.com/facebookresearch/segment-anything"
SAM_SOURCE_REPO_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

with SourceAsRoot(
    SAM_SOURCE_REPO,
    SAM_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
):
    from segment_anything.modeling.image_encoder import Attention as SAMEncoderAttention
    from segment_anything.modeling.image_encoder import (
        MLPBlock as SAMTransformerMLPBlock,
    )
    from segment_anything.modeling.image_encoder import get_rel_pos
    from segment_anything.modeling.mask_decoder import MLP as SAMMaskDecoderMLP
    from segment_anything.modeling.mask_decoder import MaskDecoder as SAMMaskDecoder
    from segment_anything.modeling.transformer import Attention as SAMDecoderAttention


def window_partition_5d(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    ---
    Lifted from segment_anything.modeling.image_encoder.window_partition
    Modified by Qualcomm to work in 5D rather than 6D.
    ---

    Partition into non-overlapping windows with padding if needed.

    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # -- Begin Qualcomm Modification --
    x = x.reshape(B * Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size, window_size, C)
    )
    # -- End Qualcomm Modification --
    return windows, (Hp, Wp)


def window_unpartition_5d(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """
    ---
    Lifted from segment_anything.modeling.image_encoder.window_unpartition
    Modified by Qualcomm to work in 5D rather than 6D.
    ---

    Window unpartition into original sequences and removing padding.

    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    # -- Begin Qualcomm Modification --
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, Hp, Wp, -1)
    # -- End Qualcomm Modification --

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


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


class SplitHeadSAMEncoderAttention(nn.Module):
    """
    SAM Attention block with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
    """

    def __init__(self, attention_block: SAMEncoderAttention):
        super().__init__()
        self.out_feature, self.in_feature = (
            attention_block.qkv.weight.shape[0] // 3 // attention_block.num_heads,
            attention_block.qkv.weight.shape[1],
        )
        chunk_size = attention_block.qkv.weight.shape[0] // 3

        bias = attention_block.qkv.bias[: self.out_feature] is not None
        self.q = torch.nn.ModuleList()
        self.k = torch.nn.ModuleList()
        self.v = torch.nn.ModuleList()
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.use_rel_pos = attention_block.use_rel_pos
        self.scale = attention_block.scale
        self.num_heads = attention_block.num_heads
        self.rel_pos_h = attention_block.rel_pos_h
        self.rel_pos_w = attention_block.rel_pos_w

        for i in range(attention_block.num_heads):
            for chunk, projList in enumerate([self.q, self.k, self.v]):
                split_layer = Conv2DInplaceLinear(
                    self.in_feature, self.out_feature, has_bias=bias
                )
                split_layer.conv2d.weight.data.copy_(
                    attention_block.qkv.weight[
                        i * self.out_feature
                        + (chunk * chunk_size) : (i + 1) * self.out_feature
                        + (chunk * chunk_size),
                        :,
                        None,
                        None,
                    ]
                )

                split_layer.conv2d.bias.data.copy_(
                    attention_block.qkv.bias[
                        i * self.out_feature
                        + (chunk * chunk_size) : (i + 1) * self.out_feature
                        + (chunk * chunk_size)
                    ]
                )

                projList.append(split_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        """
        #original code
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q0, k0, v0 = qkv.reshape(3, B * self.self.num_heads, H * W, -1).unbind(0)
        """
        x_list: list[torch.Tensor] = []
        for i in range(self.num_heads):
            q_i = self.q[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            k_i = self.k[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            v_i = self.v[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            attn_i = (q_i * self.scale) @ k_i.transpose(-2, -1)

            if self.use_rel_pos:
                attn_i = SplitHeadSAMEncoderAttention.add_decomposed_rel_pos_unpack(
                    attn_i,
                    q_i,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    (H, W),
                    (H, W),
                )

            attn_i = attn_i.softmax(dim=-1)
            x_i = (attn_i @ v_i).reshape(B, 1, H * W, -1)
            x_list.append(x_i)
        x = (
            torch.concat(x_list, dim=1)
            .reshape(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x

    @staticmethod
    def einsum_to_matmul_bhwc_hkc_bhwk(r_q, Rh):
        Rh = torch.transpose(Rh, 2, 1)
        op = torch.matmul(r_q, Rh)
        return op

    @staticmethod
    def einsum_to_matmul_bhwc_wkc_bhwk(r_q, Rw):
        r_q = torch.transpose(r_q, 2, 1)
        Rw = torch.transpose(Rw, 2, 1)
        test_result_second = torch.matmul(r_q, Rw)
        op = torch.transpose(test_result_second, 2, 1)
        return op

    @staticmethod
    def add_decomposed_rel_pos_unpack(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        ---
        Lifted from segment_anything.modeling.image_encoder.add_decomposed_rel_pos
        Modifications by Qualcomm:
         * Enable compatibility of Q shape with other changes that unpack attention QKV
         * Replace Einsum with equivalent ops (einsum is not supported by QNN)
        ---

        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h, q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

        # -- Begin Qualcomm Change
        B, _, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = SplitHeadSAMEncoderAttention.einsum_to_matmul_bhwc_hkc_bhwk(r_q, Rh)
        rel_w = SplitHeadSAMEncoderAttention.einsum_to_matmul_bhwc_wkc_bhwk(r_q, Rw)
        # -- End Qualcomm Change

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, 1, q_h * q_w, k_h * k_w)

        return attn


class SplitHeadSAMDecoderAttention(nn.Module):
    def __init__(self, attention_block: SAMDecoderAttention):
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


def sam_decoder_predict_masks(
    self: SAMMaskDecoder,
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


class Conv2DInplaceLinearSAMTransformerMLPBlock(nn.Module):
    """
    SAM MLPBlock that uses 1x1 Conv2D in place of linear layers.
    """

    def __init__(self, mlp_block: SAMTransformerMLPBlock):
        super().__init__()
        self.lin1 = Conv2DInplaceLinear.from_linear(mlp_block.lin1)
        self.lin2 = Conv2DInplaceLinear.from_linear(mlp_block.lin2)
        self.act = mlp_block.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Conv2DInplaceLinearSAMMaskDecoderMLP(nn.Module):
    """
    SAM MLP that uses 1x1 Conv2D in place of linear layers.
    """

    def __init__(self, mlp: SAMMaskDecoderMLP):
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
