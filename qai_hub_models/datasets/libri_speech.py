# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import torch
import torchaudio
from torch.nn import functional as F

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

LIBRISPEECH_FOLDER_NAME = "librispeech"
LIBRISPEECH_VERSION = 1
# LibriSpeech test-clean dataset from: www.openslr.org/12 (test-clean.tar.gz)
LIBRISPEECH_CLEAN_ASSET = CachedWebDatasetAsset.from_asset_store(
    LIBRISPEECH_FOLDER_NAME,
    LIBRISPEECH_VERSION,
    "test-clean.tar.gz",
)
DEFAULT_SEQUENCE_LENGTH = 160000  # 5 seconds at 16kHz


class LibriSpeechDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        target_sample_rate: int = 16000,
        max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        max_text_length: int = 200,
    ):
        self.base_path = LIBRISPEECH_CLEAN_ASSET.path(extracted=True).parent
        BaseDataset.__init__(self, self.base_path, split)
        self.target_sample_rate = target_sample_rate
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): The index of the audio file and transcription in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - audio (torch.Tensor): Processed audio tensor of shape [sequence_length],
                normalized and padded/truncated to max_sequence_length.
                - gt_tensor (torch.Tensor): Tensor of shape [max_text_length] containing
                ASCII character codes for the transcription, padded with zeros if needed.
        """
        # Load audio file
        audio_path = self.audio_files[index]
        audio, audio_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if audio_sr != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, audio_sr, self.target_sample_rate
            )

        # Truncate to FIXED length
        audio_len = min(audio.shape[-1], self.max_sequence_length)
        audio = audio[..., :audio_len]

        if audio_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - audio_len
            audio = F.pad(audio, (0, pad_len), mode="constant")

        audio = audio.squeeze(0)
        audio = audio / torch.max(torch.abs(audio))
        # Process transcription text with FIXED maximum length
        text = str(self.transcriptions[index])
        text = text.replace("'", "")  # Remove single quotes
        text = text[: self.max_text_length]  # Truncate to max length

        # Convert to tensor of character indices with padding
        gt_tensor = torch.tensor([ord(c) for c in text], dtype=torch.int32)

        # Pad to max_text_length
        if len(gt_tensor) < self.max_text_length:
            pad_len = self.max_text_length - len(gt_tensor)
            gt_tensor = F.pad(gt_tensor, (0, pad_len), value=0)  # 0 = padding index

        return audio, gt_tensor

    def _validate_data(self) -> bool:
        dataset_path = self.base_path / "LibriSpeech" / "test-clean"
        self.audio_files = []
        self.transcriptions = []

        # Check if dataset path exists
        if not dataset_path.exists():
            return False

        # Iterate through speaker and chapter directories
        for speaker_dir in sorted(dataset_path.glob("*")):
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.glob("*")):
                if not chapter_dir.is_dir():
                    continue

                # Load transcriptions from trans.txt
                trans_file = (
                    chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                )
                if not trans_file.exists():
                    continue

                trans_dict = {}
                with open(trans_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)  # Split on first space
                        if len(parts) == 2:
                            audio_id = parts[0].strip()
                            transcription = parts[1].strip().lower()
                            trans_dict[audio_id] = transcription

                # Collect audio files and pair with transcriptions
                for audio_path in sorted(chapter_dir.glob("*.flac")):
                    audio_id = audio_path.stem
                    if audio_id in trans_dict:
                        self.audio_files.append(audio_path)
                        self.transcriptions.append(trans_dict[audio_id])

        # Verify collected data
        if not self.audio_files or len(self.audio_files) != len(self.transcriptions):
            raise ValueError("no audio files or transciptions found")

        return True

    def __len__(self):
        return len(self.audio_files)

    def _download_data(self) -> None:
        LIBRISPEECH_CLEAN_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 500

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.openslr.org/12",
            split_description="test split",
        )
