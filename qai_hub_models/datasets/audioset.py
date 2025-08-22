# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torchaudio

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models.yamnet.app import preprocessing_yamnet_from_source
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

AUDIOSET_FOLDER_NAME = "audioset"
AUDIOSET_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 96000  # 0.96s at 16kHz
DEFAULT_NUM_CLASSES = 521
FRAME_DURATION = 0.96  # Duration of one frame in seconds

# originally comes from https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval08.tar,
AUDIOSET_ASSET = CachedWebDatasetAsset.from_asset_store(
    AUDIOSET_FOLDER_NAME,
    AUDIOSET_VERSION,
    "eval.tar",
)
# originally comes from http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
AUDIOSET_CSV = CachedWebDatasetAsset.from_asset_store(
    AUDIOSET_FOLDER_NAME,
    AUDIOSET_VERSION,
    "eval_segments.csv",
)
# originally comes from https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
AUDIOSET_CLASS_MAP = CachedWebDatasetAsset.from_asset_store(
    AUDIOSET_FOLDER_NAME, AUDIOSET_VERSION, "class_map.csv"
)


class AudioSetDataset(BaseDataset):
    """
    Initialize AudioSet dataset for Audio classification tasks.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        target_sample_rate: int = 16000,
        max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):

        self.csv_path = AUDIOSET_CSV.path()
        self.audio_dir = AUDIOSET_ASSET.path(extracted=True).parent
        self.class_map_path = AUDIOSET_CLASS_MAP.path()
        self.num_classes = DEFAULT_NUM_CLASSES
        BaseDataset.__init__(self, self.audio_dir, split)
        self.target_sample_rate = target_sample_rate
        self.max_sequence_length = max_sequence_length

    def _validate_data(self) -> bool:
        """Validate and load dataset.

        Returns:
            bool: True if data is valid and loaded successfully, False otherwise
        """
        self.audio_files: list = []
        self.labels: list = []
        self.id_to_idx: dict[str, int] = {}
        self.sample_ids: list[str] = []
        self.csv_data = None
        self.audio_frames: dict[str, int] = {}

        if not self.audio_dir.exists():
            return False

        if not self.id_to_idx:
            self._load_class_map()

        if not self.csv_data:
            self._prepare_data()

        if not self.audio_files:
            raise ValueError("No valid audio files found")

        return True

    def __getitem__(self, index) -> tuple[Any, tuple[Any, Any]]:
        """Get a single audio sample and its labels by index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            tuple: (audio_tensor, (label_tensor, sample_id)) where:
                - audio_tensor: Preprocessed audio tensor of shape (96, 64)
                - label_tensor: Multi-hot encoded labels tensor of shape (num_classes,)
                - sample_id: Original sample ID from the dataset
        """

        audio_file = self.audio_files[index]
        audio, sr = torchaudio.load(self.audio_files[index])
        if audio.shape[0] > 1:  # converts to mono
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=self.target_sample_rate
            )
        patches, _ = preprocessing_yamnet_from_source(audio)
        frame_idx = index % self.audio_frames[audio_file.stem]
        input_tensor = torch.zeros(1, 96, 64)
        if patches.shape[0] > 0 and frame_idx < patches.shape[0]:
            input_tensor = patches[frame_idx : frame_idx + 1].squeeze(0)
        return input_tensor, (self.labels[index], self.sample_ids[index])

    def _parse_audioset_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(
                self.csv_path,
                header=None,
                skiprows=3,
                skipinitialspace=True,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
        except Exception as e:
            raise ValueError(f"Failed to parse CSV {self.csv_path}: {str(e)}")

    def _load_class_map(self) -> None:
        """Load class ID to index mapping from AudioSet ontology."""
        ontology_data = pd.read_csv(self.class_map_path)
        self.id_to_idx = {
            row["mid"]: row["index"] for _, row in ontology_data.iterrows()
        }

    def _prepare_data(self) -> None:
        """Prepare data by matching audio files with CSV entries."""

        # Load CSV data once and create a lookup dictionary
        self.csv_data = self._parse_audioset_csv()

        csv_lookup = {row["YTID"]: row for _, row in self.csv_data.iterrows()}  # type: ignore

        self.audio_dir = self.audio_dir / "audio" / "eval"

        # Find all audio files
        audio_files = list(self.audio_dir.glob("*.flac"))

        for audio_file in audio_files:
            ytid = audio_file.stem
            if ytid in csv_lookup:
                row = csv_lookup[ytid]
                # Calculate duration from CSV
                duration = row["end_seconds"] - row["start_seconds"]
                # Number of frames based on audio duration
                frames_per_sample = max(1, int(duration / FRAME_DURATION))
                self.audio_frames[ytid] = frames_per_sample
                label_ids = row["positive_labels"].split(",")
                label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
                for label_id in label_ids:
                    if label_id in self.id_to_idx:
                        label_vector[self.id_to_idx[label_id]] = 1.0
                # Add each frame as a separate sample
                for frame_idx in range(frames_per_sample):
                    self.audio_files.append(audio_file)
                    self.labels.append(label_vector)
                    self.sample_ids.append(ytid)

        if not self.audio_files:
            raise ValueError("No matching audio files found for CSV entries")

    def __len__(self):
        return len(self.audio_files)

    def _download_data(self):
        AUDIOSET_CLASS_MAP.fetch()
        AUDIOSET_CSV.fetch()
        AUDIOSET_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job():
        return 11356
