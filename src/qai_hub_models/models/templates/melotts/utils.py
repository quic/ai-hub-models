# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil

import unidic
from platformdirs import user_cache_path
from unidic.download import download_version as _download_unidic

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.melotts.meloTTS_metadata_json import (
    create_tts_metadata,
)
from qai_hub_models.models.templates.voiceai_tts.tts_metadata import (
    write_tts_supplementary_files,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

UNIDIC_CACHE_PATH = user_cache_path("unidic")
UNICODE_DATA_ASSET = CachedWebModelAsset(
    url="https://www.unicode.org/Public/3.0-Update/UnicodeData-3.0.0.txt",
    model_id="melottstemplates",
    model_asset_version=1,
    filename="UnicodeData-3.0.0.txt",
)


def download_unidic() -> None:
    """
    Downloads supporting files for the unidic package to a shared global cache location.

    The default location dumps these files directly into the python environment folder,
    which makes working with multiple environments tedious, since we need to re-download
    500+mb of supporting files for every new python env.
    """
    if not os.path.exists(unidic.DICDIR):
        if os.name != "nt":
            if not os.path.exists(UNIDIC_CACHE_PATH):
                # This will delete the unidic folder in the python env if it exists already.
                # So we call it first before moving the results and symlinking the original dst folder.
                _download_unidic()
                UNIDIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(unidic.DICDIR, UNIDIC_CACHE_PATH)
            os.symlink(UNIDIC_CACHE_PATH, unidic.DICDIR)
        else:
            # Do nothing special on Windows since symlinking works poorly there.
            _download_unidic()

    try:
        import MeCab

        MeCab.Tagger()
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load TTS language data. Try deleting {unidic.DICDIR} and try again."
        ) from e
    except ImportError as e:
        raise ImportError(
            "MeloTTS is not installed correctly. Refer to the model README for installation instructions."
        ) from e


def write_melotts_supplementary_files(
    language: TTSLanguage,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    sample_rate: int,
) -> None:
    """
    Write supplementary files for MeloTTS models.

    Parameters
    ----------
    language
        Language key (e.g., "ENGLISH", "SPANISH", "CHINESE")
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info
    sample_rate
        Audio sample rate in Hz
    """
    write_tts_supplementary_files(
        language=language,
        output_dir=output_dir,
        metadata=metadata,
        create_metadata_fn=create_tts_metadata,
        sample_rate=sample_rate,
        metadata_description="TTS metadata JSON",
    )
