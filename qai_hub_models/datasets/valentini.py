# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import torchaudio
from torch.nn import functional as F

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

VALENTINI_FOLDER_NAME = "valentini"
VALENTINI_VERSION = 1
# Originally from:
# https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip
# https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip
VALENTINI_CLEAN_ASSET = CachedWebDatasetAsset.from_asset_store(
    VALENTINI_FOLDER_NAME,
    VALENTINI_VERSION,
    "clean_testset_wav.zip",
)
VALENTINI_NOISY_ASSET = CachedWebDatasetAsset.from_asset_store(
    VALENTINI_FOLDER_NAME,
    VALENTINI_VERSION,
    "noisy_testset_wav.zip",
)
DEFAULT_SEQUENCE_LENGTH = 100000


class ValentiniDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        target_sample_rate: int = 16000,
        max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        BaseDataset.__init__(self, VALENTINI_CLEAN_ASSET.path().parent, split)

        self.noisy_dir = VALENTINI_NOISY_ASSET.path() / "noisy_testset_wav"
        self.clean_dir = VALENTINI_CLEAN_ASSET.path() / "clean_testset_wav"
        self.target_sample_rate = target_sample_rate
        self.max_sequence_length = max_sequence_length

        # Get paired audio files
        self.file_pairs = []
        for noisy_file in sorted(self.noisy_dir.glob("*.wav")):
            clean_file = self.clean_dir / noisy_file.name
            if clean_file.exists():
                self.file_pairs.append((noisy_file, clean_file))

        if not self.file_pairs:
            raise ValueError("No valid noisy/clean audio pairs found")

    def __getitem__(self, index):
        noisy_file, clean_file = self.file_pairs[index]

        # Load audio files
        noisy, noisy_sr = torchaudio.load(noisy_file)
        clean, clean_sr = torchaudio.load(clean_file)

        # Resample from 48 hz to 16 hz
        if noisy_sr != self.target_sample_rate:
            noisy = torchaudio.functional.resample(
                noisy, noisy_sr, self.target_sample_rate
            )
        if clean_sr != self.target_sample_rate:
            clean = torchaudio.functional.resample(
                clean, clean_sr, self.target_sample_rate
            )

        min_len = min(noisy.shape[-1], clean.shape[-1], self.max_sequence_length)

        # Keep sequences the same length, using only data up to max_sequence_length; anything extra is left out
        start_idx = 0
        noisy = noisy[..., start_idx : start_idx + min_len]
        clean = clean[..., start_idx : start_idx + min_len]

        if min_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - min_len
            noisy = F.pad(noisy, (0, pad_len), mode="constant")
            clean = F.pad(clean, (0, pad_len), mode="constant")

        return noisy, clean

    def __len__(self):
        return len(self.file_pairs)

    def _download_data(self) -> None:
        VALENTINI_CLEAN_ASSET.fetch(extract=True)
        VALENTINI_NOISY_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/datasets/muhmagdy/valentini-noisy",
            split_description="test split",
        )
