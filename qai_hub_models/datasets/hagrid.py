# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.models._shared.mediapipe.utils import preprocess_hand_x64
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_image
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

DATASET_ID = "hagrid"
DATASET_VERSION = 1
IMAGES_DIR_NAME = "images"  # full frames (palm detector)
ROI_DIR_NAME = "rois"  # cropped hand images (landmarks)
ANNOTATIONS_NAME = "landmarks_dump.csv"

HAGRID_CLEAN_ASSET = CachedWebDatasetAsset.from_asset_store(
    DATASET_ID,
    DATASET_VERSION,
    "data.zip",
)


def _parse_lr(val: Any) -> int:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("right", "r", "1", "true", "t"):
            return 1
        if v in ("left", "l", "0", "false", "f"):
            return 0
    if isinstance(val, bool):
        return 1 if val else 0
    raise ValueError(f"Unsupported handedness value: {val!r}")


def _extract_landmarks_from_row(row: pd.Series) -> np.ndarray:
    lm_vals: list[float] = []
    for i in range(21):
        for axis in ("x", "y", "z"):
            key = f"lm_{i}_{axis}"
            if key not in row:
                raise KeyError(f"Missing column '{key}' in CSV row.")
            lm_vals.append(float(row[key]))
    return np.array(lm_vals, dtype=np.float32).reshape(21, 3)


# ===========================================================
# Palm detection dataset (shared base)
# ===========================================================
class PalmDetectorDataset(BaseDataset):
    """
    Wrapper class for mediapipe palm detection dataset

    https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_width: int = 256,
        input_height: int = 256,
    ) -> None:
        self.data_path = HAGRID_CLEAN_ASSET.path(extracted=True) / "data"
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.roi_path = self.data_path / ROI_DIR_NAME
        self.annotations_path = self.data_path / ANNOTATIONS_NAME
        self.input_width = input_width
        self.input_height = input_height
        BaseDataset.__init__(self, self.data_path, split)
        # Load annotations CSV
        self.annotations_db = self._load_annotations_db()

    # ---- shared helpers ----
    def _load_annotations_db(self) -> pd.DataFrame:
        df = pd.read_csv(str(self.annotations_path))
        required_cols = ["source_image_name", "roi_filename", "lr"] + [
            f"lm_{i}_{a}" for i in range(21) for a in ("x", "y", "z")
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing required columns: {missing}")
        return df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load a full-frame image by index, resize, and convert to tensor.

        Parameters
        ----------
        idx
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor
            Image tensor of shape (C, H, W), where:
            C = number of channels (3 for RGB)
            H = input_height (256)
            W = input_width (256)
        label
            Placeholder label (currently always 0).
        """
        row = self.annotations_db.iloc[idx]
        img_fn = str(row["source_image_name"]).strip()
        img_path = os.path.join(self.images_path, img_fn)
        image = load_image(img_path)
        image = image.resize((self.input_width, self.input_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return (image_tensor, 0)

    def __len__(self) -> int:
        try:
            return len(self.annotations_db)
        except Exception:
            return 0

    def _download_data(self) -> None:
        """Download and set up dataset."""
        HAGRID_CLEAN_ASSET.fetch(extract=True)

    def _validate_data(self) -> bool:
        return self.images_path.exists() and len(os.listdir(self.images_path)) >= 100

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "hagrid_palmdetector"


class HandLandmarkDataset(PalmDetectorDataset):
    """Wrapper class for mediapipe hand landmark dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        split
            Whether to use the train, val, or test split of the dataset.
        input_spec
            Specification object provided by model.
        """
        if input_spec is not None:
            input_width = input_spec["image"][0][-1]
            input_height = input_spec["image"][0][-2]
        else:
            # Default ROI network input size 224x224
            input_width = 224
            input_height = 224
        super().__init__(split, input_width, input_height)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load a cropped hand ROI image by index, resize, and convert to tensor.

        Parameters
        ----------
        idx
            The index of the sample to retrieve.

        Returns
        -------
        roi_tensor
            Image tensor of shape (C, H, W), where:
            C = number of channels (3 for RGB)
            H = input_height (224)
            W = input_width (224)
        label
            Placeholder label (currently always 0).
        """
        row = self.annotations_db.iloc[idx]
        roi_fn = str(row["roi_filename"]).strip()
        roi_path = os.path.join(self.roi_path, roi_fn)
        image = load_image(roi_path)
        image = image.resize((self.input_width, self.input_height))
        roi_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return (roi_tensor, 0)

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "hagrid_handlandmark"


class GestureClassifierDataset(PalmDetectorDataset):
    """

    Wrapper class for mediapipe gesture embedder and classifier dataset

    Dataset that yields preprocessed 64-D inputs for the gesture embedder/classifier:
      - x64_a: landmarks (63) + handedness (1) for the original hand
      - x64_b: mirrored variant (x-axis flipped) with handedness inverted

    """

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], str]:
        """
        Return a tuple containing input hand and mirrored hand tensors and gesture label.

        Parameters
        ----------
        idx
            The index used to retrieve the sample from the dataset.

        Returns
        -------
        x64_a
            Tensor of shape (64,) with normalized features from original landmarks + handedness.
        x64_b
            Tensor of shape (64,) with normalized features from x-mirrored landmarks + inverted handedness.
        gesture_label
            One of: "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
            "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou".
        """
        row = self.annotations_db.iloc[idx]
        lm_np = _extract_landmarks_from_row(row)
        landmarks_tensor = torch.from_numpy(lm_np.astype(np.float32))
        # Handedness scalar (1,) in {0.0, 1.0}
        lr_val = _parse_lr(row["lr"])
        lr_tensor = torch.tensor([float(lr_val)], dtype=torch.float32)
        hand = landmarks_tensor.unsqueeze(0)
        lr = lr_tensor.unsqueeze(0)
        # Branch A: original
        x64_a = preprocess_hand_x64(hand, lr, mirror=False).squeeze(0)
        # Branch B: mirrored
        x64_b = preprocess_hand_x64(hand, lr, mirror=True).squeeze(0)
        gesture_label = str(row.get("gesture_label", ""))
        return ((x64_a, x64_b), gesture_label)

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "hagrid_gesturerecognizer"
