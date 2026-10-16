# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
MeloTTS (Text-to-Speech) metadata schema.

This module defines the structure for ``tts.json`` files that document
MeloTTS model capabilities, voice specifications, runtime parameters,
and asset locations.
"""

from __future__ import annotations

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.voiceai_tts.tts_metadata import TTSMetadata
from qai_hub_models.utils.base_config import BaseQAIHMConfig


class ModelAssets(BaseQAIHMConfig):
    """Paths to model asset files."""

    bert_model: str | None = None
    bert_tokenizer: str | None = None
    bert_normalizer: str | None = None
    melo_encoder: str | None = None
    melo_sdp: str | None = None
    melo_flow: str | None = None
    melo_decoder: str | None = None
    g2p_encoder: str | None = None
    g2p_decoder: str | None = None


def create_tts_metadata(
    language: TTSLanguage,
    tokenizer_bin_name: str | None,
    normalizer_bin_name: str,
    metadata: ModelMetadata,
    sample_rate: int,
) -> TTSMetadata:
    """
    Generate ``TTSMetadata`` for a MeloTTS model.

    Parameters
    ----------
    language
        Language code (e.g., ``ENGLISH``, ``SPANISH``, ``CHINESE``).
    tokenizer_bin_name
        Name of the tokenizer binary file.
    normalizer_bin_name
        Name of the normalizer binary file.
    metadata
        ``ModelMetadata`` instance containing model files and tool versions.
    sample_rate
        Audio sample rate in Hz.

    Returns
    -------
    TTSMetadata
        The generated TTS metadata object.
    """
    assets = ModelAssets()
    for file_name in metadata.model_files:
        lower = file_name.lower()
        if "charsiu_encoder" in lower:
            assets.g2p_encoder = file_name
        elif "charsiu_decoder" in lower:
            assets.g2p_decoder = file_name
        elif "bert" in lower:
            assets.bert_model = file_name
        elif "encoder" in lower:
            assets.melo_encoder = file_name
        elif "sdp" in lower:
            assets.melo_sdp = file_name
        elif "decoder" in lower:
            assets.melo_decoder = file_name
        elif "flow" in lower:
            assets.melo_flow = file_name

    assets.bert_tokenizer = tokenizer_bin_name
    assets.bert_normalizer = normalizer_bin_name

    return TTSMetadata.from_tts_model(
        model_type="melo",
        display_model_name="MeloTTS",
        language=language,
        tool_versions=metadata.tool_versions,
        assets=assets,
        sample_rate=sample_rate,
        is_model_quantized=True,
    )
