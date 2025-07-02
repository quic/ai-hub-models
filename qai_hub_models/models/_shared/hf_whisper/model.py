# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

import torch
from qai_hub.client import Device
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from qai_hub_models.models._shared.hf_whisper.model_adaptation import (
    QcWhisperDecoder,
    QcWhisperEncoder,
    monkey_patch_model,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "hf_whisper_asr_shared"
MODEL_ASSET_VERSION = 1

# 20ms sample rate
SAMPLE_RATE = 16000

# Audio chunk length in seconds
CHUNK_LENGTH = 30

# Samples per chunk
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 20ms samples in a 30-second chunk

# The official default max decoded length is 448. We use decoded length 200 for benchmarking purpose
MEAN_DECODE_LEN = 200

# Audio embedding length 1500
AUDIO_EMB_LEN = 1500

# Audio length per MEL feature
MELS_AUDIO_LEN = AUDIO_EMB_LEN * 2

# Mask neg
MASK_NEG = -100.0


class HfWhisperEncoder(BaseModel):
    """
    HfWhisperEncoder optimized for export and inference.

    It takes audio input (mel) and directly produce cross attention
    kv-cache.
    """

    def __init__(
        self, config: WhisperConfig, model: QcWhisperEncoder | None = None
    ) -> None:
        super().__init__()
        self.encoder = model
        self.config = config

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Return cross attention key and value cache tensors
        assert self.encoder is not None, "model is None"
        kv_cache_cross = self.encoder(input_features)[0]
        return kv_cache_cross

    @staticmethod
    def get_input_spec(num_mel_bin: int = 80) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return dict(input_features=((1, num_mel_bin, MELS_AUDIO_LEN), "float32"))

    def _get_input_spec_for_instance(self) -> InputSpec:
        return self.__class__.get_input_spec(
            self.config.num_mel_bins,
        )

    @staticmethod
    def get_output_names(
        num_blocks: int = 12,
    ) -> list[str]:
        return [
            f"{prefix}_cache_cross_{i}"
            for i in range(num_blocks)
            for prefix in ("k", "v")
        ]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.config.decoder_layers)

    @classmethod
    def from_pretrained(cls):
        hf_whisper = HfWhisper.from_pretrained()
        return cls(hf_whisper.config, hf_whisper.encoder)

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
        if (
            precision == Precision.float
            and target_runtime.compilation_uses_qnn_converters
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class HfWhisperDecoder(BaseModel):
    """
    HfWhisperDecoder optimized for export and inference:

    Wraps `HfWhisperDecoder` to facilitate export.
    """

    def __init__(
        self, config: WhisperConfig, model: QcWhisperDecoder | None = None
    ) -> None:
        super().__init__()
        self.decoder = model
        self.config = config

    @property
    def num_blocks(self) -> int:
        return self.config.decoder_layers

    def forward(
        self,
        *args: Any,
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:

        - input_ids: torch.tensor, shape = (batch_size, <= n_ctx)
            the text tokens

        - attention_mask: torch.tensor, shape = (1, 1, 1, 200)
            Mask to avoid performing attention on padding token indices.

        - kv_caches
          k_cache_self_{i}_in: key cache for self attention:
          [num_heads, 1, attn_dim/num_heads, self.max_decode_len - 1]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output
          v_cache_self_{i}_in: value cache for self attention:
          [num_heads, 1, self.max_decode_len - 1, attn_dim/num_heads]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output
          k_cache_cross_{i}: key cache for cross attention:
          [num_heads, 1, attn_dim/num_heads, AUDIO_EMB_LEN]
          v_cache_cross_{i}: value cache for cross attention:
          [num_heads, 1, AUDIO_EMB_LEN, attn_dim/num_heads]

        - position_ids: torch.tensor, shape = (1)
            index to get the positional encoding for x.

        Returns:

        - logits: of shape [1, 51865, 1, 1]
        - kv_cache_self_new: updated key value cache for self attention
        """
        assert self.decoder is not None, "model is None"
        input_ids = args[0]
        attention_mask = args[1]
        kv_caches = args[2:-1]
        kv_cache_self = [
            (kv_caches[i], kv_caches[i + 1]) for i in range(0, self.num_blocks * 2, 2)
        ]
        kv_cache_cross = [
            (kv_caches[i], kv_caches[i + 1])
            for i in range(
                self.num_blocks * 2,
                self.num_blocks * 4,
                2,
            )
        ]
        position_ids = args[-1]

        logits, kv_cache_self_new = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache_self,
            cross_attn_past_key_value=kv_cache_cross,
            position_ids=position_ids,
        )
        return logits, kv_cache_self_new

    @staticmethod
    def get_input_spec(
        num_blocks: int = 12,
        attention_dim: int = 768,
        num_heads: int = 12,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        specs = dict(
            input_ids=((1, 1), "int32"),
            attention_mask=((1, 1, 1, MEAN_DECODE_LEN), "float32"),
        )
        kv_cache_self = {}
        for i in range(num_blocks):
            kv_cache_self[f"k_cache_self_{i}_in"] = (
                (num_heads, 1, attention_dim // num_heads, MEAN_DECODE_LEN - 1),
                "float32",
            )
            kv_cache_self[f"v_cache_self_{i}_in"] = (
                (num_heads, 1, MEAN_DECODE_LEN - 1, attention_dim // num_heads),
                "float32",
            )
        kv_cache_cross = {}
        for i in range(num_blocks):
            kv_cache_cross[f"k_cache_cross_{i}"] = (
                (num_heads, 1, attention_dim // num_heads, AUDIO_EMB_LEN),
                "float32",
            )
            kv_cache_cross[f"v_cache_cross_{i}"] = (
                (num_heads, 1, AUDIO_EMB_LEN, attention_dim // num_heads),
                "float32",
            )
        specs.update(kv_cache_self)
        specs.update(kv_cache_cross)
        specs["position_ids"] = ((1,), "int32")

        return specs

    def _get_input_spec_for_instance(self) -> InputSpec:
        return self.__class__.get_input_spec(
            self.config.decoder_layers,
            self.config.d_model,
            self.config.decoder_attention_heads,
        )

    @staticmethod
    def get_output_names(
        num_blocks: int = 12,
    ) -> list[str]:
        return ["logits"] + [
            f"{prefix}_cache_self_{i}_out"
            for i in range(num_blocks)
            for prefix in ("k", "v")
        ]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.num_blocks)

    @classmethod
    def from_pretrained(cls):
        hf_whisper = HfWhisper.from_pretrained()
        return cls(hf_whisper.config, hf_whisper.decoder)

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
        if (
            precision == Precision.float
            and target_runtime.compilation_uses_qnn_converters
        ):
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options


class HfWhisper(CollectionModel):
    def __init__(
        self,
        encoder: HfWhisperEncoder,
        decoder: HfWhisperDecoder,
        config: WhisperConfig,
        hf_source: str,
    ) -> None:
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.hf_source = hf_source

    @classmethod
    @abstractmethod
    def get_hf_whisper_version(cls) -> str:
        pass

    @classmethod
    def from_pretrained(cls):
        hf_whisper_version = cls.get_hf_whisper_version()
        orig_whisper = WhisperForConditionalGeneration.from_pretrained(
            hf_whisper_version
        )
        orig_whisper.config.return_dict = False
        orig_whisper.config.tie_word_embeddings = False
        orig_whisper.config.mask_neg = MASK_NEG

        whisper_model = orig_whisper.model
        monkey_patch_model(whisper_model)

        encoder = HfWhisperEncoder(orig_whisper.config, whisper_model.get_encoder())
        decoder = HfWhisperDecoder(orig_whisper.config, whisper_model.get_decoder())

        encoder.eval()
        decoder.eval()
        return cls(
            encoder,
            decoder,
            orig_whisper.config,
            hf_whisper_version,
        )


def get_feature_extractor(
    hf_whisper_version: str = "openai/whisper-small",
) -> WhisperFeatureExtractor:
    """
    feature_extractor to use for Whisper
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(hf_whisper_version)
    return feature_extractor


def get_tokenizer(hf_whisper_version: str = "openai/whisper-small") -> WhisperTokenizer:
    """
    Tokenizer to use for Whisper
    """
    tokenizer = WhisperTokenizer.from_pretrained(hf_whisper_version)
    return tokenizer
