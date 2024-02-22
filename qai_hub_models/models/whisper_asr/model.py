# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import whisper  # type: ignore

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MAX_DECODE_LEN = 448

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
MEL_FILTER_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "openai_assets/mel_filters.npz"
)


class Whisper(CollectionModel):
    def __init__(
        self,
        encoder: Callable[[torch.Tensor], List[torch.Tensor]],
        decoder: Callable[..., Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]],
        num_decoder_blocks: int,
        attention_dim: int,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.num_decoder_blocks = num_decoder_blocks
        self.attention_dim = attention_dim

    @classmethod
    def from_pretrained(cls, model: str = "tiny.en"):
        # For other model sizes, see https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
        return cls.from_source_model(whisper.load_model(model))

    @classmethod
    def from_source_model(cls, whisper_model: Any):
        encoder = WhisperEncoderInf(whisper_model)
        decoder = WhisperDecoderInf(whisper_model.decoder)
        num_decoder_blocks = len(decoder.blocks)
        attention_dim = decoder.attention_dim
        return cls(encoder, decoder, num_decoder_blocks, attention_dim)  # type: ignore


class WhisperEncoderInf(BaseModel):
    """
    WhisperEncoder optimized for export and inference.

    It takes audio input (mel) and directly produce cross attention
    kv-cache.
    """

    def __init__(self, model: whisper.model.Whisper):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        # Return 2 * self.num_blocks tensors (k, v for each block)
        encoder_out = self.model.encoder(audio)
        res = []
        for residual_block in self.model.decoder.blocks:
            res.append(residual_block.cross_attn.key(encoder_out))
            res.append(residual_block.cross_attn.value(encoder_out))
        return res

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return dict(audio=((1, 80, 3000), "float32"))

    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained().encoder


class WhisperDecoderInf(BaseModel):
    """
    whisper.model.TextDecoder optimized for export and inference:

    Wraps `whisper.model.TextDecoder` to facilitate export:

    1. kv cache inputs are individual tensors instead of a list of tensors
    2. kv cache inputs are required, not optional
    """

    def __init__(self, model: whisper.model.TextDecoder):
        super().__init__()
        assert isinstance(model, whisper.model.TextDecoder)

        # Wraps `ResidualAttentionBlock` in
        # `ResidualAttentionBlockWrapper`
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlockWrapper(b) for b in model.blocks]
        )
        for m in ["token_embedding", "ln"]:
            self.add_module(m, getattr(model, m))
        for p in ["positional_embedding"]:
            self.register_parameter(p, getattr(model, p))

    @property
    def attention_dim(self):
        return self.blocks[0].attn_ln.weight.shape[0]

    def forward(self, x: torch.Tensor, *kv_cache_args, **kv_cache_kwargs):
        """
        Args:

        - x: torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens

        - kv_cache_args: Tuple of length 4 * num_decoder_blocks. Elements are:

            b{i}_cross_attn_k: [1, 1500, attn_dim]
            b{i}_cross_attn_v: [1, 1500, attn_dim]

            for i = 0, ..., num_blocks

            followed by

            b{i}_self_attn_k: [1, decoded_len, attn_dim]
            b{i}_self_attn_v: [1, decoded_len, attn_dim]

            for i = 0, ..., num_blocks

        Returns:

        - logits: of shape [1, 1, 51864]
        - b0_self_attn_k, b0_self_attn_v, b1_self_attn_k, ...: Updated self attn cache.
          2*num_decoder_blocks
        """
        if not kv_cache_args:
            kv_cache_args = list(kv_cache_kwargs.values())
        assert isinstance(self.token_embedding, torch.nn.Module)  # for mypy
        assert isinstance(self.ln, torch.nn.Module)  # for mypy
        assert isinstance(self.positional_embedding, torch.nn.Parameter)  # for mypy
        # Set up kv_cache
        kv_cache = {}  # torch.nn.Module -> torch.Tensor
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            kv_cache.update(
                {
                    block.attn.key: kv_cache_args[2 * num_blocks + i * 2],
                    block.attn.value: kv_cache_args[2 * num_blocks + i * 2 + 1],
                    block.cross_attn.key: kv_cache_args[i * 2],
                    block.cross_attn.value: kv_cache_args[i * 2 + 1],
                }
            )
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        # x shape: (1, 1, 384)
        kv_cache_new = []
        for block in self.blocks:
            x, k_cache, v_cache = block(x, kv_cache=kv_cache)
            kv_cache_new.append(k_cache.float())
            kv_cache_new.append(v_cache.float())

        x = self.ln(x)
        logits = (
            x
            @ torch.transpose(
                self.token_embedding.weight.to(x.dtype), 0, 1  # type: ignore
            )
        ).float()

        # shape: [1, 1, 51864]
        return (logits,) + tuple(kv_cache_new)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        specs = dict(x=((1, 1), "int32"))
        for i in range(len(self.blocks)):
            specs[f"b{i}_cross_attn_k"] = ((1, 1500, self.attention_dim), "float32")
            specs[f"b{i}_cross_attn_v"] = ((1, 1500, self.attention_dim), "float32")

        # Use mean length for profiling
        mean_decode_len = MAX_DECODE_LEN // 2

        for i in range(len(self.blocks)):
            specs[f"b{i}_self_attn_k"] = (
                (1, mean_decode_len, self.attention_dim),
                "float32",
            )
            specs[f"b{i}_self_attn_v"] = (
                (1, mean_decode_len, self.attention_dim),
                "float32",
            )

        return specs

    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained().decoder


class MHAWrapper(torch.nn.Module):
    """
    Wrapper around whisper.model.MultiHeadAttention to leverage kv cache for
    efficient inference. The original whisper.model.MultiHeadAttention doesn't
    returns the updated kv cache but relies on pytorch hook which
    cannot be exported for on-device inference. This wrapper fixes that.

    If attn_type == "self_attention", the kv cache is updated before they are returned.

    If attn_type == "cross_attention", the kv cache is returned without any update.

    Note that unlike whisper.model.MultiHeadAttention, this wrapper is
    optimized for inference so it doesn't take mask as an input.
    """

    def __init__(self, model: whisper.model.MultiHeadAttention, attn_type: str):
        """
        attn_type: one of {"self_attention", "cross_attention"}
        """
        super().__init__()
        assert isinstance(model, whisper.model.MultiHeadAttention)
        self.attn_type = attn_type
        self.n_head = model.n_head
        for m in ["query", "key", "value", "out"]:
            self.add_module(m, getattr(model, m))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Dict[torch.nn.Module, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:

        - x: shape [1, 1, attention_dim]. Input feature.

        - kv_cache: 4 * num_decoder_blocks entries representing self attention
          and cross attention from all attention blocks. Each entry of shape
          [1, decoded_len, attention_dim]. We'd only use cache relevant to this
          particular attention layer and ignore other entries in the dict.

        Returns:

        - x_out: attention output

        - updated k, v cache: of shape [1, decoded_len+1, attention_dim]
        """
        assert isinstance(self.query, torch.nn.Module)  # for mypy
        assert isinstance(self.key, torch.nn.Module)  # for mypy
        assert isinstance(self.value, torch.nn.Module)  # for mypy
        assert isinstance(self.out, torch.nn.Module)  # for mypy
        q = self.query(x)

        if self.attn_type == "self_attention":
            k_cache = kv_cache[self.key]
            v_cache = kv_cache[self.value]
            k = self.key(x)
            v = self.value(x)
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        else:  # cross_attention
            k, v = kv_cache[self.key], kv_cache[self.value]

        wv = qkv_attention(q, k, v, self.n_head)
        # Return updated kv cache
        return self.out(wv), k.detach(), v.detach()


def qkv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_head: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adapted from whisper.model.MultiHeadAttention.qkv_attention
    """
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // n_head) ** -0.25
    q = q.view(*q.shape[:2], n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.view(*k.shape[:2], n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.view(*v.shape[:2], n_head, -1).permute(0, 2, 1, 3)

    qk = q @ k
    if mask is not None:
        qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlockWrapper(torch.nn.Module):
    """
    Wrapper around whisper.model.ResidualAttentionBlock to leverage kv cache
    for efficient inference. The original whisper.model.ResidiualAttentionBlock
    doesn't returns the updated kv cache but relies on pytorch hook which
    cannot be exported for on-device inference. This wrapper fixes that.
    """

    def __init__(self, model: whisper.model.ResidualAttentionBlock):
        super().__init__()
        assert isinstance(model, whisper.model.ResidualAttentionBlock)
        # Wraps `MultiheadAttention` to `MultiheadAttentionWrapper`
        self.attn = MHAWrapper(model.attn, "self_attention")
        self.cross_attn = MHAWrapper(model.cross_attn, "cross_attention")
        for m in ["attn_ln", "cross_attn_ln", "mlp", "mlp_ln"]:
            self.add_module(m, getattr(model, m))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Dict[torch.nn.Module, torch.Tensor],
    ):
        """
        Args: Same as MHAWrapper
        Returns: Same as MHAWrapper
        """
        # Get updated self attention kv cache
        assert isinstance(self.attn, torch.nn.Module)  # for mypy
        assert isinstance(self.attn_ln, torch.nn.Module)  # for mypy
        assert isinstance(self.cross_attn_ln, torch.nn.Module)  # for mypy
        assert isinstance(self.cross_attn, torch.nn.Module)  # for mypy
        assert isinstance(self.mlp, torch.nn.Module)  # for mypy
        assert isinstance(self.mlp_ln, torch.nn.Module)  # for mypy
        x_attn, k_cache, v_cache = self.attn(self.attn_ln(x), kv_cache=kv_cache)
        x = x + x_attn
        if self.cross_attn:
            # Ignore cross attn kv cache which is constant (pre-computed in
            # `WhisperCrossAttnKVCacheTorch`)
            x_cross_attn, _, _ = self.cross_attn(
                self.cross_attn_ln(x), kv_cache=kv_cache
            )
            x = x + x_cross_attn
        x = x + self.mlp(self.mlp_ln(x))
        return x, k_cache, v_cache
