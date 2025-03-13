# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch
import whisper
from qai_hub.client import Device

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "whisper_asr_shared"
MODEL_ASSET_VERSION = 1

# 20ms sample rate
SAMPLE_RATE = 16000

# Length of the Hann window signal used when applying a FFT to the audio.
N_FFT = 400

# Number of audio samples between adjacent STFT columns when applying FFT to the audio.
HOP_LENGTH = 160

# Audio chunk length in seconds
CHUNK_LENGTH = 30

# Samples per chunk
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 20ms samples in a 30-second chunk

# The official default max decoded length is 448. We use mean decoded length 224 for benchmarking purpose
MEAN_DECODE_LEN = 224

# MEL filter to be applied to audio after applying FFT
MEL_FILTER_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "openai_assets/mel_filters.npz"
)

# The number of Mel features per audio context
N_MELS = 80

# Audio embedding length
AUDIO_EMB_LEN = int(N_SAMPLES / N_MELS / 4)

# Audio length per MEL feature
MELS_AUDIO_LEN = AUDIO_EMB_LEN * 2


class Whisper(CollectionModel):
    def __init__(
        self,
        encoder: Callable[[torch.Tensor], list[torch.Tensor]],
        decoder: Callable[..., tuple[torch.Tensor, tuple[torch.Tensor, ...]]],
        num_decoder_blocks: int,
        attention_dim: int,
        num_heads: int,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.num_decoder_blocks = num_decoder_blocks
        self.attention_dim = attention_dim
        self.num_decoder_heads = num_heads
        self.mean_decode_len = MEAN_DECODE_LEN

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
        num_heads = decoder.num_heads
        return cls(encoder, decoder, num_decoder_blocks, attention_dim, num_heads)


class WhisperEncoderInf(BaseModel):
    """
    WhisperEncoder optimized for export and inference.

    It takes audio input (mel) and directly produce cross attention
    kv-cache.
    """

    def __init__(self, model: whisper.model.Whisper):
        super().__init__()
        self.encoder = model.encoder
        dims = model.dims

        states_per_head = dims.n_audio_state // dims.n_audio_head
        scale = states_per_head**-0.25

        self.cross_attn_key_list = torch.nn.ModuleList(
            [
                SplitLinear(block.cross_attn.key, dims.n_audio_head, scale)
                for block in model.decoder.blocks
            ]
        )
        self.cross_attn_value_list = torch.nn.ModuleList(
            [
                SplitLinear(block.cross_attn.value, dims.n_audio_head)
                for block in model.decoder.blocks
            ]
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return cross attention key and value cache tensors
        encoder_out = self.encoder(audio)
        k_cache = torch.cat(
            [
                key(encoder_out, transpose=True).unsqueeze(0)
                for key in self.cross_attn_key_list
            ],
            dim=0,
        )
        v_cache = torch.cat(
            [value(encoder_out).unsqueeze(0) for value in self.cross_attn_value_list],
            dim=0,
        )
        return k_cache, v_cache

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return dict(audio=((1, N_MELS, MELS_AUDIO_LEN), "float32"))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["k_cache", "v_cache"]

    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained().encoder

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--compute_unit" not in profile_options
        ):
            profile_options = profile_options + " --compute_unit gpu"
        return profile_options + " --max_profiler_iterations 10"

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
        if precision == Precision.float and target_runtime in {
            TargetRuntime.QNN,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        }:
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class WhisperDecoderInf(BaseModel):
    """
    whisper.model.TextDecoder optimized for export and inference:

    Wraps `whisper.model.TextDecoder` to facilitate export:

    1. kv cache inputs are individual tensors instead of a list of tensors
    2. kv cache inputs are required, not optional
    """

    def __init__(
        self, model: whisper.model.TextDecoder, max_decode_len: int = MEAN_DECODE_LEN
    ):
        super().__init__()
        assert isinstance(model, whisper.model.TextDecoder)

        self.max_decode_len = max_decode_len

        # Wraps `ResidualAttentionBlock` in
        # `ResidualAttentionBlockWrapper`
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlockWrapper(b) for b in model.blocks]
        )

        for m in ["token_embedding", "ln"]:
            self.add_module(m, getattr(model, m))

        # Replace `whisper.model.TextDecoder.positional_embedding` (nn.Parameter) with nn.Embedding for easier lookup
        self.positional_embedding = torch.nn.Embedding(
            max_decode_len, self.token_embedding.weight.shape[1]
        )
        self.positional_embedding.weight = torch.nn.Parameter(
            model.positional_embedding[:max_decode_len, :]
        )

        self.logits = torch.nn.Linear(
            self.token_embedding.weight.shape[1],
            self.token_embedding.weight.shape[0],
            bias=False,
        )
        self.logits.weight = self.token_embedding.weight

        # Since kv cache is a fixed size, mask out elements
        # that correspond to not yet used entries.
        # The kv cache for current token is inserted at the last
        # index, with the previous cache shifted down by one element.
        self.mask = torch.nn.Embedding(max_decode_len, max_decode_len)
        mask = torch.zeros([max_decode_len, max_decode_len], dtype=torch.float32)
        for c_idx in range(0, max_decode_len):
            mask[c_idx, 0 : max_decode_len - c_idx - 1] = -100
        self.mask.weight = torch.nn.Parameter(mask)

    @property
    def attention_dim(self):
        return self.blocks[0].attn_ln.weight.shape[0]

    @property
    def num_heads(self):
        return self.blocks[0].attn.n_head

    @property
    def num_blocks(self):
        return len(self.blocks)

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        k_cache_cross: torch.Tensor,
        v_cache_cross: torch.Tensor,
        k_cache_self: torch.Tensor,
        v_cache_self: torch.Tensor,
    ):
        """
        Args:

        - x: torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens

        - index: torch.tensor, shape = (1, 1)
            index to get the positional encoding for x.

        - k_cache_cross: key cache for cross attention:
          [num_blocks, num_heads, attn_dim/num_heads, AUDIO_EMB_LEN]

        - v_cache_cross: value cache for cross attention:
          [num_blocks, num_heads, AUDIO_EMB_LEN, attn_dim/num_heads]

        - k_cache_self: key cache for self attention:
          [num_blocks, num_heads, attn_dim/num_heads, self.max_decode_len]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output

        - v_cache_self: value cache for self attention:
          [num_blocks, num_heads, self.max_decode_len, attn_dim/num_heads]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output

        Returns:

        - logits: of shape [1, 1, 51864]
        - k_cache_self_new: updated key cache for self attention
        - v_cache_self_new: updated value cache for self attention
        """

        assert isinstance(self.token_embedding, torch.nn.Module)  # for mypy
        assert isinstance(self.ln, torch.nn.Module)  # for mypy
        assert isinstance(self.positional_embedding, torch.nn.Embedding)  # for mypy
        # Set up kv_cache
        kv_cache = {}  # torch.nn.Module -> torch.Tensor
        for i, block in enumerate(self.blocks):
            kv_cache.update(
                {
                    block.attn.key: k_cache_self[i : i + 1],
                    block.attn.value: v_cache_self[i : i + 1],
                    block.cross_attn.key: k_cache_cross[i : i + 1],
                    block.cross_attn.value: v_cache_cross[i : i + 1],
                }
            )

        x = self.token_embedding(x) + self.positional_embedding(index)
        mask = self.mask(index)

        # x shape: (1, 1, 384)
        k_cache_new = []
        v_cache_new = []
        for block_idx in range(self.num_blocks):
            x, k_cache, v_cache = self.blocks[block_idx](x, mask, kv_cache=kv_cache)
            k_cache_new.append(k_cache.float())
            v_cache_new.append(v_cache.float())

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        logits = self.logits(x).float()

        return logits, torch.cat(k_cache_new), torch.cat(v_cache_new)

    @staticmethod
    def get_input_spec(
        num_blocks: int, attention_dim: int, num_heads: int
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        specs: InputSpec = dict(
            x=((1, 1), "int32"),
            index=((1, 1), "int32"),
            k_cache_cross=(
                (num_blocks, num_heads, attention_dim // num_heads, AUDIO_EMB_LEN),
                "float32",
            ),
            v_cache_cross=(
                (num_blocks, num_heads, AUDIO_EMB_LEN, attention_dim // num_heads),
                "float32",
            ),
            k_cache_self=(
                (num_blocks, num_heads, attention_dim // num_heads, MEAN_DECODE_LEN),
                "float32",
            ),
            v_cache_self=(
                (num_blocks, num_heads, MEAN_DECODE_LEN, attention_dim // num_heads),
                "float32",
            ),
        )

        return specs

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logits", "k_cache", "v_cache"]

    def _get_input_spec_for_instance(self) -> InputSpec:
        return self.__class__.get_input_spec(
            len(self.blocks), self.attention_dim, self.num_heads
        )

    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained().decoder

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
        if precision == Precision.float and target_runtime in {
            TargetRuntime.QNN,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        }:
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class SplitLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module, num_splits: int, scale: float = 1.0):
        """
        Split Linear operation into multiple instances
        Multi-head cross attention
        Uses pre-computed cross kv cache passed as input to the
        decoder model
        """
        super().__init__()
        weight = linear.weight
        has_bias = False if linear.bias is None else True
        if has_bias:
            bias = linear.bias.reshape(num_splits, -1) * scale
        split_weight = weight.reshape(num_splits, -1, weight.shape[1]) * scale
        self.split_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    split_weight.shape[1], split_weight.shape[2], bias=has_bias
                )
                for split_idx in range(num_splits)
            ]
        )
        for split_idx in range(num_splits):
            self.split_linears[split_idx].weight = torch.nn.Parameter(
                split_weight[split_idx, :, :]
            )
            if has_bias:
                self.split_linears[split_idx].bias = torch.nn.Parameter(bias[split_idx])

    def forward(self, x: torch.Tensor, transpose: bool = False):
        """
        produces output with dimension
        [num_splits, input rows, output_features / num_splits]
        If transpose is True, will transpose last two indices
        """
        if transpose:
            x = torch.cat(
                [spl(x).transpose(-1, -2) for spl in self.split_linears], dim=-3
            )
        else:
            x = torch.cat([spl(x) for spl in self.split_linears], dim=-3)
        return x


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
        mask: torch.Tensor,
        kv_cache: dict[torch.nn.Module, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:

        - x: shape [1, 1, attention_dim]. Input feature.

        - kv_cache: 4 * num_decoder_blocks entries representing self attention
          and cross attention from all attention blocks. Each k entry of shape
          [1, num_heads, attention_dim // num_heads, context_len] and
          each v entry of shape
          [1, num_heads, context_len, attention_dim // num_heads].
          We'd only use cache relevant to this particular attention layer
          and ignore other entries in the dict.

        Returns:

        - x_out: attention output

        - updated k, v cache: with same shape as input
        """
        assert isinstance(self.query, torch.nn.Module)  # for mypy
        assert isinstance(self.key, torch.nn.Module)  # for mypy
        assert isinstance(self.value, torch.nn.Module)  # for mypy
        assert isinstance(self.out, torch.nn.Module)  # for mypy
        q = self.query(x)
        q = q.view(q.shape[0], self.n_head, 1, -1)
        if self.attn_type == "self_attention":
            k_cache = kv_cache[self.key]
            v_cache = kv_cache[self.value]
            k = self.key(x).unsqueeze(3)
            k = k.view(k.shape[0], self.n_head, -1, 1)
            v = self.value(x).unsqueeze(2)
            v = v.view(k.shape[0], self.n_head, 1, -1)
            # shift kv cache and insert new k and v entries
            k = torch.cat((k_cache[:, :, :, 1:], k), dim=-1)
            v = torch.cat((v_cache[:, :, 1:, :], v), dim=-2)
            wv = qkv_attention(q, k, v, self.n_head, mask=mask)
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
    wv_list = []
    # Split heads in qkv calculation
    for h in range(n_head):
        qk = q[:, h : h + 1, :, :] @ k[:, h : h + 1, :, :]
        if mask is not None:
            qk = qk + mask
        w = torch.nn.functional.softmax(qk, dim=-1)
        wv_list.append(w @ v[:, h : h + 1, :, :])
    wv = torch.cat(wv_list, dim=1)
    wv = wv.view(wv.shape[0], 1, -1)
    return wv


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

        states_per_head = model.attn.query.weight.shape[0] // model.attn.n_head
        scale = states_per_head**-0.25
        self.cross_attn = MHAWrapper(model.cross_attn, "cross_attention")

        # Apply scale for qkv to parameters
        with torch.no_grad():
            self.attn.query.weight *= scale
            self.attn.query.bias *= scale
            self.attn.key.weight *= scale
            self.cross_attn.query.weight *= scale
            self.cross_attn.query.bias *= scale
            self.cross_attn.key.weight *= scale

        for m in ["attn_ln", "cross_attn_ln", "mlp", "mlp_ln"]:
            self.add_module(m, getattr(model, m))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: dict[torch.nn.Module, torch.Tensor],
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
        x_attn, k_cache, v_cache = self.attn(
            self.attn_ln(x), mask=mask, kv_cache=kv_cache
        )
        x = x + x_attn
        if self.cross_attn:
            # Ignore cross attn kv cache which is constant (pre-computed in
            # `WhisperCrossAttnKVCacheTorch`)
            x_cross_attn, _, _ = self.cross_attn(
                self.cross_attn_ln(x), mask=mask, kv_cache=kv_cache
            )
            x = x + x_cross_attn
        x = x + self.mlp(self.mlp_ln(x))
        return x, k_cache, v_cache
