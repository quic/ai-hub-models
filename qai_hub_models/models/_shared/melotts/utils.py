# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import shutil

import unidic
from platformdirs import user_cache_path
from unidic.download import download_version as _download_unidic

UNIDIC_CACHE_PATH = user_cache_path("unidic")


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
