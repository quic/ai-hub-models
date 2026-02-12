# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from qai_hub_models.utils.asset_loaders import SourceAsRoot

SOURCE_REPO = "https://github.com/k2-fsa/icefall"
COMMIT_HASH = "693f069de73fd91d7c2009571245d97221cc3a3f"

with SourceAsRoot(
    SOURCE_REPO,
    COMMIT_HASH,
    "icefall",
    1,
):
    sys.path.append("egs/librispeech/ASR/pruned_transducer_stateless7_streaming")
    os.system(
        "cp egs/librispeech/ASR/pruned_transducer_stateless7/scaling.py    egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
    )
    from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.scaling import (
        penalize_abs_values_gt,
        softmax,
    )
    from egs.librispeech.ASR.pruned_transducer_stateless7_streaming.zipformer import (
        AttentionDownsample,
        ConvolutionModule,
        RelPositionalEncoding,
        RelPositionMultiheadAttention,
    )


# ============================================================================
# Qc Module Classes
# ============================================================================


class QcRelPositionMultiheadAttention(nn.Module):
    """
    Replace original RelPositionMultiheadAttention of zipformer in the icefall repository.
    Main changes: replace nn.functional.linear with nn.Linear and torch.cat optimization.
    """

    def __init__(self, orig_module: RelPositionMultiheadAttention) -> None:
        super().__init__()
        self.embed_dim = orig_module.embed_dim
        self.attention_dim = orig_module.attention_dim
        self.num_heads = orig_module.num_heads
        self.head_dim = orig_module.head_dim
        self.pos_dim = orig_module.pos_dim

        self.in_proj = orig_module.in_proj
        self.linear_pos = orig_module.linear_pos
        self.out_proj = orig_module.out_proj
        self.in_proj2 = orig_module.in_proj2
        self.out_proj2 = orig_module.out_proj2

        # Replace functional linear with nn.Linear
        out_features, in_features = orig_module.out_proj.weight.shape
        self.out_proj_linear = nn.Linear(in_features, out_features, bias=True)
        self.out_proj_linear.weight.data.copy_(orig_module.out_proj.weight.data)
        self.out_proj_linear.bias.data.copy_(orig_module.out_proj.bias.data)

    def prepare_pos_emb(self, seq_len: int, embed_dim: int) -> None:
        pos_emb_len = int(6 * seq_len - 1)
        pos_emb = np.fromfile(
            f"pos_emb/pos_emb_{pos_emb_len}.bin", dtype=np.float32
        ).reshape(1, -1, embed_dim)
        pos_emb_tensor = torch.tensor(pos_emb)
        pos_emb_tensor = pos_emb_tensor.to(self.linear_pos.weight.device)
        self.register_buffer("pos_emb", pos_emb_tensor, persistent=True)
        self.pos_emb = self.linear_pos(pos_emb_tensor)

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            x,
            weights,
            cached_key,
            cached_val,
        ) = self.streaming_multi_head_attention_forward(
            self.in_proj(x),
            self.attention_dim,
            self.num_heads,
            cached_key=cached_key,
            cached_val=cached_val,
        )
        return x, weights, cached_key, cached_val

    def streaming_multi_head_attention_forward(
        self,
        x_proj: Tensor,
        attention_dim: int,
        num_heads: int,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        seq_len, bsz, _ = x_proj.size()
        head_dim = attention_dim // num_heads
        pos_dim = self.pos_dim

        # self-attention
        q = x_proj[..., 0:attention_dim]
        k = x_proj[..., attention_dim : 2 * attention_dim]
        value_dim = attention_dim // 2
        v = x_proj[..., 2 * attention_dim : 2 * attention_dim + value_dim]
        p = x_proj[..., 2 * attention_dim + value_dim :]

        left_context_len = cached_key.shape[0]

        ## ---------- For qnn convert op(start) ---------- ##
        # Dynamic lengths
        T_new = k.shape[0]  # current step length

        # Pure functional rewrite:
        # - Pad k and v on the left by left_context_len zeros along time dimension
        # - Pad cached_key and cached_val on the right by T_new zeros
        # - Add them to emulate "overwrite prefix" without slice writes
        k = F.pad(k, (0, 0, 0, 0, left_context_len, 0), value=0.0) + F.pad(
            cached_key, (0, 0, 0, 0, 0, T_new), value=0.0
        )

        v = F.pad(v, (0, 0, 0, 0, left_context_len, 0), value=0.0) + F.pad(
            cached_val, (0, 0, 0, 0, 0, T_new), value=0.0
        )
        ## ---------- For qnn convert op(end) ---------- ##

        # Update cached contexts
        cached_key = k[-left_context_len:, ...]
        cached_val = v[-left_context_len:, ...]

        kv_len = k.shape[0]
        q = q.reshape(seq_len, bsz, num_heads, head_dim)
        p = p.reshape(seq_len, bsz, num_heads, pos_dim)
        k = k.reshape(kv_len, bsz, num_heads, head_dim)
        v = v.reshape(kv_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        q = q.permute(1, 2, 0, 3)
        p = p.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 3, 0)

        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos = self.pos_emb.reshape(1, seq_len2, num_heads, pos_dim).permute(0, 2, 3, 1)
        pos_weights = torch.matmul(p, pos)

        if torch.jit.is_tracing():
            (batch_size, num_heads, time1, n) = pos_weights.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(kv_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_weights = pos_weights.reshape(-1, n)
            pos_weights = torch.gather(pos_weights, dim=1, index=indexes)
            pos_weights = pos_weights.reshape(batch_size, num_heads, time1, kv_len)
        else:
            pos_weights = pos_weights.as_strided(
                (bsz, num_heads, seq_len, kv_len),
                (
                    pos_weights.stride(0),
                    pos_weights.stride(1),
                    pos_weights.stride(2) - pos_weights.stride(3),
                    pos_weights.stride(3),
                ),
                storage_offset=pos_weights.stride(3) * (seq_len - 1),
            )

        attn_output_weights = torch.matmul(q, k) + pos_weights
        attn_output_weights = attn_output_weights.view(bsz * num_heads, seq_len, kv_len)
        attn_output_weights = softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, attention_dim // 2)
        )
        attn_output = self.out_proj_linear(attn_output)

        return attn_output, attn_output_weights, cached_key, cached_val

    def streaming_forward2(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_heads = self.num_heads
        (seq_len, bsz, _) = x.shape
        head_dim = self.attention_dim // num_heads
        v = self.in_proj2(x)

        left_context_len = cached_val.shape[0]

        ## ---------- For qnn convert op(start) ---------- ##
        # Dynamic lengths
        T1 = cached_val.shape[0]  # length of cached prefix (time dimension)
        T_new = v.shape[0]  # length of new segment to append

        # Pure functional concat using Pad + Add (no ScatterND / no Concat):
        # - Pad v on the left by T1 zeros along the time dimension
        # - Pad cached_val on the right by T_new zeros
        # - Add them to form [T1 + T_new, ...] where the prefix is cached_val and the tail is v
        v = F.pad(v, (0, 0, 0, 0, T1, 0), value=0.0) + F.pad(
            cached_val, (0, 0, 0, 0, 0, T_new), value=0.0
        )
        ## ---------- For qnn convert op(end) ---------- ##

        cached_val = v[-left_context_len:]
        seq_len2 = left_context_len + seq_len
        v = v.reshape(seq_len2, bsz * num_heads, head_dim // 2).transpose(0, 1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, self.attention_dim // 2)
        )
        return self.out_proj2(attn_output), cached_val


class QcConvolutionModule(nn.Module):
    """
    Replace original ConvolutionModule of zipformer.
    Main changes: 1D conv -> 2D conv and torch.cat optimization.
    """

    def __init__(self, orig_module: ConvolutionModule) -> None:
        super().__init__()
        self.deriv_balancer1 = orig_module.deriv_balancer1
        self.lorder = orig_module.lorder
        self.deriv_balancer2 = orig_module.deriv_balancer2
        self.activation = orig_module.activation

        # Convert 1D convolutions to 2D convolutions
        orig_conv1 = orig_module.pointwise_conv1
        self.pointwise_conv1 = nn.Conv2d(
            in_channels=orig_conv1.in_channels,
            out_channels=orig_conv1.out_channels,
            kernel_size=(1, orig_conv1.kernel_size[0]),
            stride=(1, orig_conv1.stride[0]),
            padding=(0, orig_conv1.padding[0]),
            bias=orig_conv1.bias is not None,
        )
        self.pointwise_conv1.weight.data = orig_conv1.weight.data.unsqueeze(2)
        if orig_conv1.bias is not None:
            self.pointwise_conv1.bias = orig_conv1.bias

        orig_depthwise = orig_module.depthwise_conv
        self.depthwise_conv = nn.Conv2d(
            in_channels=orig_depthwise.in_channels,
            out_channels=orig_depthwise.out_channels,
            kernel_size=(1, orig_depthwise.kernel_size[0]),
            stride=(1, orig_depthwise.stride[0]),
            padding=(0, orig_depthwise.padding[0]),
            groups=orig_depthwise.groups,
            bias=orig_depthwise.bias is not None,
        )
        self.depthwise_conv.weight.data = orig_depthwise.weight.data.unsqueeze(2)
        if orig_depthwise.bias is not None:
            self.depthwise_conv.bias = orig_depthwise.bias

        orig_conv2 = orig_module.pointwise_conv2
        if hasattr(orig_conv2, "conv"):
            # ScaledConv1d case
            actual_conv = orig_conv2.conv
            self.pointwise_conv2 = nn.Conv2d(
                in_channels=actual_conv.in_channels,
                out_channels=actual_conv.out_channels,
                kernel_size=(1, actual_conv.kernel_size[0]),
                stride=(1, actual_conv.stride[0]),
                padding=(0, actual_conv.padding[0]),
                bias=actual_conv.bias is not None,
            )
            self.pointwise_conv2.weight.data = actual_conv.weight.data.unsqueeze(2)
            if actual_conv.bias is not None:
                self.pointwise_conv2.bias = actual_conv.bias
        else:
            # Regular Conv1d case
            self.pointwise_conv2 = nn.Conv2d(
                in_channels=orig_conv2.in_channels,
                out_channels=orig_conv2.out_channels,
                kernel_size=(1, orig_conv2.kernel_size[0]),
                stride=(1, orig_conv2.stride[0]),
                padding=(0, orig_conv2.padding[0]),
                bias=orig_conv2.bias is not None,
            )
            self.pointwise_conv2.weight.data = orig_conv2.weight.data.unsqueeze(2)
            if orig_conv2.bias is not None:
                self.pointwise_conv2.bias = orig_conv2.bias

    def streaming_forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        x = x.permute(1, 2, 0)  # (batch, channels, time)
        x = x.unsqueeze(2)  # (batch, channels, 1, time)

        x = self.pointwise_conv1(x)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)

        ## ---------- For qnn convert op(start) ---------- ##
        # Expand cache to 4D: [N, C, 1, W_cache]
        cache_4d = cache.unsqueeze(2)

        # Dynamic widths
        W_cache = cache_4d.shape[3]  # prefix width to overwrite
        W_x = x.shape[3]  # original x width

        # Pure functional rewrite:
        # - Pad x on the left by W_cache zeros along the last (W) dimension
        # - Pad cache_4d on the right by W_x zeros to match total width
        # - Add them to emulate "overwrite prefix" without slice writes
        x = F.pad(x, (W_cache, 0), value=0.0) + F.pad(cache_4d, (0, W_x), value=0.0)
        ## ---------- For qnn convert op(end) ---------- ##

        # Update cache
        cache = x[:, :, 0, -self.lorder :]
        x = self.depthwise_conv(x)
        x = self.deriv_balancer2(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = x.squeeze(2)  # (batch, channel, time)
        return x.permute(2, 0, 1), cache


class QcAttentionDownsample(nn.Module):
    """
    Replace original AttentionDownsample of zipformer.
    Main changes: replace sum operations with manual loops.
    """

    def __init__(self, orig_module: AttentionDownsample) -> None:
        super().__init__()
        self.query = orig_module.query
        self.downsample = orig_module.downsample
        self.extra_proj = orig_module.extra_proj

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, 1, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds, (src.shape[0], d_seq_len, ds)

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        # scores = (src * self.query).sum(dim=-1, keepdim=True)
        scores = torch.matmul(src, self.query.unsqueeze(-1))

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

        weights = scores.softmax(dim=1)

        # ans1 is the first `in_channels` channels of the output
        # ans = (src * weights).sum(dim=1)
        src_reshaped = src.squeeze(2)  # [16, 2, 384]
        weights_reshaped = weights.squeeze(2)  # [16, 2, 1]
        weights_transposed = weights_reshaped.transpose(1, 2)  # [16, 1, 2]
        ans = torch.bmm(weights_transposed, src_reshaped)

        src = src.permute(0, 2, 1, 3).reshape(d_seq_len, batch_size, ds * in_channels)

        if self.extra_proj is not None:
            ans2 = self.extra_proj(src)
            ans = torch.cat((ans, ans2), dim=2)
        return ans


# ============================================================================
# Helper Classes
# ============================================================================


class DualInt8LinearConv(nn.Module):
    """
    Quantization-friendly linear layer using dual INT8 weights + INT16 activations.

    This module implements a linear transformation using two separate INT8 quantized
    weight matrices (conv0, conv1) with different scales (s0, s1) to improve the
    representable range of weights. Activations are quantized to INT16 for better
    precision while maintaining hardware efficiency.

    Key features:
    - Dual INT8 weight quantization: W = s0*q0 + s1*q1 (where q0, q1 are INT8)
    - INT16 activation quantization with softplus smoothing
    - Hardware-friendly 1x1 Conv2d implementation
    - Expects 4D input tensors (N, C, 1, 1) for compatibility with Conv2DWrapper
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Dual INT8 weight convolutions (frozen parameters for quantized inference)
        self.conv0 = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

        # Freeze quantized weight parameters
        for p in self.conv0.parameters():
            p.requires_grad = False
        for p in self.conv1.parameters():
            p.requires_grad = False

        # Quantization scales for dual INT8 weights
        self.register_buffer("s0", torch.ones(out_features))  # Primary scale
        self.register_buffer("s1", torch.zeros(out_features))  # Residual scale
        self.register_buffer(
            "wq0_colsum", torch.zeros(out_features)
        )  # For optimization
        self.register_buffer(
            "wq1_colsum", torch.zeros(out_features)
        )  # For optimization

        # Optional bias parameter
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        # INT16 activation quantization range
        self.qmin_x = -(2 ** (16 - 1))  # -32768
        self.qmax_x = (2 ** (16 - 1)) - 1  # 32767

    def _quantize_act_int16(
        self, x2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use softplus instead of clamp to avoid hard threshold errors
        amax = torch.amax(torch.abs(x2d), dim=1, keepdim=True)
        safe_amax = torch.nn.functional.softplus(amax)  # Smooth to avoid zero
        scale_x = safe_amax / float(self.qmax_x)

        zp_x = torch.zeros_like(scale_x)
        xq = torch.round(x2d / scale_x).clamp(self.qmin_x, self.qmax_x)
        return xq, scale_x, zp_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect 4D input: (N, Cin, 1, 1)
        N, Cin, H, W = x.shape
        assert H == 1 and W == 1, "Only 1x1 4D tensors supported"
        x2d = x.view(N, Cin)

        xq, scale_x, _ = self._quantize_act_int16(x2d)
        xq4 = xq.view(N, Cin, 1, 1)
        y0 = self.conv0(xq4)
        y1 = self.conv1(xq4)

        y_int = y0 * self.s0.view(1, -1, 1, 1) + y1 * self.s1.view(1, -1, 1, 1)
        y = y_int * scale_x.view(N, 1, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)

        return y  # Return 4D output

    @torch.no_grad()
    def prepare_from_weight(
        self, weight_4d: torch.Tensor, bias_1d: torch.Tensor | None = None
    ) -> None:
        Cout, Cin, kh, kw = weight_4d.shape
        assert kh == 1 and kw == 1
        assert Cin == self.in_features and Cout == self.out_features
        W = weight_4d.view(Cout, Cin)

        max_abs = torch.clamp(W.abs().amax(dim=1), min=1e-8)
        s0 = max_abs / 127.0
        q0 = torch.round(W / s0.unsqueeze(1)).clamp(-128, 127)

        R = W - s0.unsqueeze(1) * q0
        max_abs_R = R.abs().amax(dim=1)
        threshold = 1e-3
        s1 = torch.where(
            max_abs_R > threshold, max_abs_R / 127.0, torch.zeros_like(max_abs_R)
        )
        q1 = torch.where(
            (s1 > 0).unsqueeze(1),
            torch.round(R / s1.unsqueeze(1)).clamp(-128, 127),
            torch.zeros_like(R),
        )

        self.conv0.weight.copy_(q0.view(Cout, Cin, 1, 1))
        self.conv1.weight.copy_(q1.view(Cout, Cin, 1, 1))

        self.s0 = s0
        self.s1 = s1
        self.wq0_colsum = q0.sum(dim=1)
        self.wq1_colsum = q1.sum(dim=1)

        if self.bias is not None and bias_1d is not None:
            self.bias.copy_(bias_1d.detach())


class Conv2DWrapper(nn.Module):
    """Unified wrapper to replace nn.Linear with either nn.Conv2d or DualInt8LinearConv."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantization_friendly: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_friendly = quantization_friendly

        # if quantization_friendly:
        # Use quantized module with dual INT8 weights + INT16 activations
        #    self.linear_conv = DualInt8LinearConv(in_features, out_features, bias=bias)
        # else:
        # Use regular Conv2d
        self.linear_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )

    @torch.no_grad()
    def prepare_from_linear(self, linear: nn.Linear) -> None:
        """Sync weights from nn.Linear to conv layer."""
        assert (
            linear.in_features == self.in_features
            and linear.out_features == self.out_features
        )

        # if self.quantization_friendly:
        # Quantized path
        #    W4 = linear.weight.detach().view(self.out_features, self.in_features, 1, 1)
        #    b1 = linear.bias.detach() if linear.bias is not None else None
        #    self.linear_conv.prepare_from_weight(W4, b1)
        # else:
        # Regular Conv2d path
        self.linear_conv.weight.data = linear.weight.data.view(
            self.out_features, self.in_features, 1, 1
        )
        if linear.bias is not None:
            self.linear_conv.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = x.reshape(x.size(0), x.size(1), 1, 1)
            x = self.linear_conv(x)
            x = x.reshape(-1, x.size(1))
        elif x.dim() == 3:
            x = x.permute(1, 2, 0).unsqueeze(2)  # 0 1 2 -> 1 2 0 -> 1 2 3 0
            x = self.linear_conv(x)
            x = x.squeeze(2).permute(2, 0, 1)  # 1 2 3 0 -> 1 2 0 -> 0 1 2
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        return x


class DummyPositionalEncoding(nn.Module):
    def forward(self, src: Tensor, left_context_len: int) -> Tensor:
        return torch.zeros_like(src)


# ============================================================================
# Main Modification Functions
# ============================================================================


def Modify_EncoderModule(model: nn.Module, model_config: dict | None = None) -> None:
    """
    Replace all encoder modules with their Qc versions.

    Parameters
    ----------
    model
        The model to modify.
    model_config
        Configuration for RelPositionMultiheadAttention (optional).
    """
    # Replace all encoder modules in one pass
    _replace_encoder_modules(model)

    # Handle RelPositionMultiheadAttention specific configuration
    if model_config is not None:
        _dummy_encoder_pos(model)
        _prepare_RelPositionMultiheadAttention(model, model_config)


def replace_linear_with_conv2d(
    model: nn.Module,
    inplace: bool = True,
    exclude: list | None = None,
    quantization_friendly_modules: list | None = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with Conv2DWrapper.

    Parameters
    ----------
    model
        The model to modify.
    inplace
        Whether to modify the model in-place.
    exclude
        List of module names to exclude from replacement.
    quantization_friendly_modules
        List of module names to use quantization-friendly approach for both dim 2 and 3.

    Returns
    -------
    model : nn.Module
        The modified model with Linear layers replaced by Conv2DWrapper.
    """
    if exclude is None:
        exclude = []
    if quantization_friendly_modules is None:
        quantization_friendly_modules = []

    def replace_fn(module: nn.Module, name: str) -> nn.Module:
        if isinstance(module, nn.Linear) and name not in exclude:
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            # Determine whether to use quantization-friendly approach
            quantization_friendly = bool(
                quantization_friendly_modules and name in quantization_friendly_modules
            )

            # Create unified Conv2DWrapper with appropriate switch
            conv_wrapper = Conv2DWrapper(
                in_features,
                out_features,
                bias=bias,
                quantization_friendly=quantization_friendly,
            )
            conv_wrapper.prepare_from_linear(module)
            conv_wrapper.to(module.weight.device)
            return conv_wrapper
        return module

    if not inplace:
        model = model.__class__(**model.__dict__)

    for name, module in list(model.named_children()):
        new_module = replace_fn(module, name)
        if new_module is not module:
            setattr(model, name, new_module)
        else:
            replace_linear_with_conv2d(
                module,
                inplace=inplace,
                exclude=exclude,
                quantization_friendly_modules=quantization_friendly_modules,
            )

    return model


# ============================================================================
# Internal Helper Functions
# ============================================================================


def _replace_encoder_modules(model: nn.Module, module_list: list | None = None) -> None:
    """
    Recursively replace encoder module instances in one pass.

    Parameters
    ----------
    model
        The model to modify.
    module_list
        List of module types to replace. If None, replaces all supported modules.
        Supported types: RelPositionMultiheadAttention, ConvolutionModule, AttentionDownsample.
    """
    if module_list is None:
        module_list = [
            RelPositionMultiheadAttention,
            ConvolutionModule,
            AttentionDownsample,
        ]

    for name, module in list(model.named_children()):
        if (
            isinstance(module, RelPositionMultiheadAttention)
            and RelPositionMultiheadAttention in module_list
        ):
            new_nn = QcRelPositionMultiheadAttention(module)
            setattr(model, name, new_nn)
        elif isinstance(module, ConvolutionModule) and ConvolutionModule in module_list:
            new_nn1 = QcConvolutionModule(module)
            setattr(model, name, new_nn1)
        elif (
            isinstance(module, AttentionDownsample)
            and AttentionDownsample in module_list
        ):
            new_nn2 = QcAttentionDownsample(module)
            setattr(model, name, new_nn2)
        else:
            _replace_encoder_modules(module, module_list)


def _dummy_encoder_pos(model: nn.Module) -> None:
    """Replace encoder positional encoding with dummy version."""
    for _name, module in model.named_children():
        if hasattr(module, "encoder_pos"):
            module.encoder_pos = DummyPositionalEncoding()
        _dummy_encoder_pos(module)
    if hasattr(model, "encoder_pos"):
        model.encoder_pos = DummyPositionalEncoding()


def _prepare_RelPositionMultiheadAttention(model: Any, model_config: dict) -> None:
    """Prepare positional embeddings for RelPositionMultiheadAttention."""
    num_encoder_layers = model_config["num_encoder_layers"]
    downsampling_factor = model_config["downsampling_factor"]
    encoder_dim = model_config["encoder_dim"]
    chunk_size = model_config["decode_chunk_size"]

    _generate_positional_embedding(encoder_dim, chunk_size, downsampling_factor)

    for encoder_idx in range(len(num_encoder_layers)):
        layers = num_encoder_layers[encoder_idx]
        factor = downsampling_factor[encoder_idx]
        dim = encoder_dim[encoder_idx]
        for layer_idx in range(layers):
            if encoder_idx == 0:
                module = model.encoder.encoders[encoder_idx].layers[layer_idx].self_attn
            else:
                module = (
                    model.encoder.encoders[encoder_idx]
                    .encoder.layers[layer_idx]
                    .self_attn
                )
            seq_len = chunk_size / factor
            module.prepare_pos_emb(seq_len, dim)


def _generate_positional_embedding(
    encoder_dims: list, chunk_size: int, downsampling_factors: list
) -> None:
    """Generate positional embeddings and save to files."""
    if not os.path.isdir("pos_emb"):
        os.makedirs("pos_emb")

    seq_len = [int(chunk_size / i) for i in downsampling_factors]
    for dim, i in zip(encoder_dims, seq_len, strict=False):
        bin_path = f"pos_emb/pos_emb_{6 * i - 1}.bin"
        if os.path.exists(bin_path):
            continue
        encoder_pos = RelPositionalEncoding(dim, dropout_rate=0.1)
        encoder_pos.eval()
        x = torch.ones(i, 1, dim)
        left_context_len = 4 * i
        pos_emb = encoder_pos(x, left_context_len)
        pos_emb.detach().cpu().numpy().tofile(bin_path)
