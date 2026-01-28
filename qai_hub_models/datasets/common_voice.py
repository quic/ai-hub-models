# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.input_spec import InputSpec

CommonVoice_FOLDER_NAME = "common_voice"
CommonVoice_VERSION = 1

CommonVoice_ASSET = CachedWebDatasetAsset.from_asset_store(
    CommonVoice_FOLDER_NAME,
    CommonVoice_VERSION,
    "common_voice.zip",
)


class CommonVoiceDataset(BaseDataset):
    def __init__(
        self,
        input_tar: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        common_voice_path = (
            CommonVoice_ASSET.path(extracted=True) / CommonVoice_FOLDER_NAME
        )
        BaseDataset.__init__(self, common_voice_path, split)

        self.wav_list = []
        for root, _dirs, files in os.walk(self.dataset_path):
            for name in files:
                if name.endswith(".wav"):
                    self.wav_list.append(os.path.join(root, name))

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, index: int) -> tuple[str, list[None]]:
        """
        Returns a tuple of wav feature and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        wav_file:
            the path of a single wav file
        ground_truth:
            Empty list, no ground truth data.
        """
        return self.wav_list[index], []

    def _download_data(self) -> None:
        CommonVoice_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 200
