# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from pathlib import Path

import pandas as pd
import torch

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models._shared.video_classifier.utils import (
    get_class_name_kinetics_400,
    preprocess_video_kinetics_400,
    read_video_per_second,
    sample_video,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

KINETICS400_FOLDER_NAME = "kinetics400"
KINETICS400_VERSION = 1

# Some of the video files in the training data downloaded from the Internet
# have corrupted video files that can't be opened. If we try to load these samples
# at runtime, it throws an error and kills the process. We decided it best to remove
# these files right after downloading to avoid this error.
CORRUPTED_TRAIN_VIDEOS = [
    "-I4Ggi6-QOE_000054_000064.mp4",
    "-K_uevUt2V8_000003_000013.mp4",
    "-6wkYqjFei0_000015_000025.mp4",
    "-IT7W5_Y3Gc_000003_000013.mp4",
    "-4c4r9YeS6s_000098_000108.mp4",
    "--PyMoD3_eg_000020_000030.mp4",
    "-3pp5xan1Hw_000006_000016.mp4",
]


def _get_labeled_data(
    videos_folder: Path, labels_csv_path: Path
) -> tuple[list[str], list[int]]:
    """
    Given the folder with a subset of videos and the appropriate labels file,
    returns a list of filenames and a list of label indices for the subset that match.
    """
    video_metadata_rows = []
    for filename in os.listdir(videos_folder):
        filename_split = filename[: -len(".mp4")].split("_")
        youtube_id = "_".join(filename_split[:-2])
        start = int(filename_split[-2])
        end = int(filename_split[-1])
        video_metadata_rows.append((youtube_id, start, end))
    video_metadata_df = pd.DataFrame(
        video_metadata_rows, columns=["youtube_id", "time_start", "time_end"]
    )
    labels_df = pd.read_csv(labels_csv_path)
    join_df = labels_df.merge(
        video_metadata_df, on=["youtube_id", "time_start", "time_end"], how="inner"
    )

    # Sort to ensure deterministic ordering regardless of filesystem
    join_df = join_df.sort_values(by=["youtube_id", "time_start"])
    video_paths: list[str] = []
    label_indices: list[int] = []
    label_index_map = {
        label: i for (i, label) in enumerate(get_class_name_kinetics_400())
    }
    for _, row in join_df.iterrows():
        assert isinstance(row, pd.Series)
        video_paths.append(
            f"{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4"
        )
        label_indices.append(label_index_map[row.label])
    return video_paths, label_indices


class Kinetics400Dataset(BaseDataset):
    """
    Class for using the Kinetics400 dataset for video classification:
        https://github.com/cvdfoundation/kinetics-dataset
    """

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN, num_frames: int = 16):
        self.num_frames = num_frames
        self.split_str = split.name.lower()
        self.videos_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/{self.split_str}/part_0.tar.gz",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join(self.split_str, "part_0.tar.gz"),
        )
        self.csv_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/annotations/{self.split_str}.csv",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join("annotations", f"{self.split_str}.csv"),
        )
        self.videos_folder = self.videos_asset.path().parent
        BaseDataset.__init__(self, str(self.videos_folder), split=split)

    def __len__(self) -> int:
        # Number of non-corrupted videos in each split
        return 993 if self.split == DatasetSplit.TRAIN else 1000

    def _validate_data(self) -> bool:
        if not self.csv_asset.path().exists():
            return False

        videos_folder = self.videos_asset.path().parent
        if not videos_folder.exists():
            return False

        self.mp4_files, self.label_indices = _get_labeled_data(
            self.videos_folder, self.csv_asset.path()
        )

        if len(self.mp4_files) != len(self):
            return False

        if len(self.label_indices) != len(self):
            return False

        return True

    def preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return preprocess_video_kinetics_400(tensor)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video = read_video_per_second(str(self.videos_folder / self.mp4_files[index]))
        video = sample_video(video, self.num_frames)
        video = self.preprocess_tensor(video)

        # Type hint says output must be a tensor, but this only works correctly
        # if the raw int label is returned.
        return (video, self.label_indices[index])  # type: ignore

    def _download_data(self) -> None:
        self.videos_asset.fetch(extract=True)
        self.csv_asset.fetch()

        if self.split == DatasetSplit.TRAIN:
            for video in CORRUPTED_TRAIN_VIDEOS:
                os.remove(self.videos_folder / video)
