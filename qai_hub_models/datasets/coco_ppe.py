# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    setup_fiftyone_env,
)
from qai_hub_models.models.gear_guard_net.model import GearGuardNet
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

DEFAULT_SELECTED_LIST_FILE = (
    QAIHM_PACKAGE_ROOT.parent
    / "qai_hub_models"
    / "models"
    / "gear_guard_net"
    / "coco_2017_select.txt"
)

DATASET_ID = "coco_ppe"
DATASET_ASSET_VERSION = 1


class CocoPPEDataset(BaseDataset):
    r"""
    Subset wrapper around COCO-2017 dataset that filters samples to a provided list.

    The selection list should contain image paths within MSCOCO that include the split
    and filename, for example:
      - 'val2017/000000140556.jpg'
      - 'train2017/000000146786.jpg'
    Backslashes are supported (e.g., 'val2017\\000000140556.jpg') and will be normalized.

    Only entries that exist within the requested split (train or val of COCO-2017)
    will be kept; entries from other splits (e.g., train2014 or test2017) are ignored.

    By default, list of selected samples is defined DEFAULT_SELECTED_LIST_FILE
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        selected_list_file: str | None = None,
        selected_paths: Iterable[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split
            Whether to use the train or val split of the dataset (COCO-2017).

        input_spec
            Model input spec; determines shapes for model input produced by this dataset.

        selected_list_file
            Optional path to a text file containing one path per line that identifies
            selected images. Lines beginning with '#' are ignored. If neither this nor
            selected_paths is provided, a placeholder path is used: DEFAULT_SELECTED_LIST_FILE

        selected_paths
            Optional in-memory iterable of path strings. Useful for programmatic configuration.
        """
        # If nothing provided, use the placeholder path so users know where to plug their list
        if selected_list_file is None and selected_paths is None:
            selected_list_file = str(DEFAULT_SELECTED_LIST_FILE)

        # Store selection inputs for use during filtering
        self.selected_list_file = selected_list_file
        self.selected_paths = (
            list(selected_paths) if selected_paths is not None else None
        )

        # FiftyOne package manages dataset so pass a dummy name for data path
        BaseDataset.__init__(self, "non_existent_dir", split, input_spec)

        # input_spec is (h, w) and target_image_size is (w, h)
        input_spec = input_spec or GearGuardNet.get_input_spec()

        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.max_boxes = 1

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        tuple[
            str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ]:
        """
        Get a single sample from the dataset.

        This dataset does not contain any annotations, num_boxes of every item will be 0.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        scaled_padded_torch_image
            Preprocessed image tensor of shape (C, H, W), range [0, 1].
        metadata
            image_id
                Image identifier.
            scale_factor
                Preprocessing scale factor.
            pad
                Padding applied.
            boxes
                Bounding boxes (zeros for this dataset).
            labels
                Labels (zeros for this dataset).
            num_boxes
                Number of boxes (zero for this dataset).
        """
        from fiftyone.core.sample import SampleView

        # Open PIL image sample.
        sample: SampleView = self.dataset[index : index + 1].first()
        image = Image.open(sample.filepath).convert("RGB")

        # Convert to torch (NCHW, range [0, 1]) tensor.
        torch_image = app_to_net_image_inputs(image)[1]
        # Scale and center-pad image to user-requested target image shape.
        scaled_padded_torch_image, scale_factor, pad = resize_pad(
            torch_image, (self.target_h, self.target_w)
        )

        num_boxes = 0
        boxes = torch.zeros((self.max_boxes, 4))
        labels = torch.zeros(self.max_boxes)
        image_id = self._filepath2imageId(Path(sample.filepath).name)

        return scaled_padded_torch_image.squeeze(0), (
            image_id,
            torch.tensor([scale_factor]),
            torch.tensor([pad]),
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        num_samples
            Number of samples in the dataset.
        """
        return len(self.dataset)

    def _validate_data(self) -> bool:
        """
        Validate that the dataset has been properly loaded.

        Returns
        -------
        is_valid
            True if the dataset attribute exists, False otherwise.
        """
        return hasattr(self, "dataset")

    def _download_data(self) -> None:
        """
        Download and load the COCO dataset, filtering to the selected subset.

        This method loads the COCO-2017 dataset using FiftyOne and filters it to only
        include images specified in the selection list. The dataset is sorted by filepath
        to ensure deterministic ordering.

        Raises
        ------
        ValueError
            If no target images are found in the dataset after filtering.
        """
        # Build the allowed base filenames for this split
        split_str = self.split_str
        sel_image_ids = self._build_allowed_image_ids_for_split(split_str)

        if not sel_image_ids:
            # If no selection provided or nothing matches, keep the dataset as-is
            raise ValueError("sample_ids = 0, could not find target images in dataset")

        # Load the split using the parent implementation (sets self.dataset)
        setup_fiftyone_env()

        # This is an expensive import, so don't want to unnecessarily import it in
        # other files that import datasets/__init__.py
        import fiftyone.zoo as foz

        # Sorting by filepath ensures a deterministic ordering every time this is called
        self.dataset = foz.load_zoo_dataset(
            "coco-2017", split=split_str, shuffle=False, image_ids=sel_image_ids
        ).sort_by("filepath")

    @staticmethod
    def default_samples_per_job() -> int:
        """
        Get the default number of samples to process in each inference job.

        Returns
        -------
        samples_per_job
            Default number of samples per job (300).
        """
        return 300

    @staticmethod
    def get_dataset_metadata(split: DatasetSplit) -> DatasetMetadata:
        """
        Get metadata information about the dataset.

        Parameters
        ----------
        split
            The dataset split (TRAIN or VAL).

        Returns
        -------
        DatasetMetadata
            Metadata object containing dataset information including link and split description.

        Raises
        ------
        ValueError
            If an unsupported split is provided.
        """
        if split == DatasetSplit.TRAIN:
            split_description = "train2017 split"
        elif split == DatasetSplit.VAL:
            split_description = "val2017 split"
        else:
            raise ValueError(
                f"Unsupported dataset split: {split}. "
                "Only DatasetSplit.TRAIN and DatasetSplit.VAL are supported."
            )
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description=split_description,
        )

    def _build_allowed_image_ids_for_split(self, split_str: str) -> set[str]:
        """
        Parse selected_list_file and/or selected_paths, normalize separators, and return
        a set of image IDs that belong to the requested split.

        This method reads from the selected list file and/or selected paths, normalizes
        path separators, filters entries by the requested split, and extracts base filenames
        to create a set of allowed image IDs.

        Parameters
        ----------
        split_str
            The split string ('validation' or 'train') to filter images by.

        Returns
        -------
        allowed_image_ids
            Set of image IDs (basenames without leading zeros and .jpg extension)
            that belong to the requested split.
        """
        entries: list[str] = []

        if self.selected_list_file:
            try:
                with open(self.selected_list_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        entries.append(line)
            except Exception:
                # If the file cannot be read, fall back to selected_paths only
                pass

        if self.selected_paths:
            entries.extend(self.selected_paths)

        # Normalize path separators and filter by split
        split_token = "val2017" if split_str == "validation" else "train2017"
        allowed: set[str] = set()
        for p in entries:
            norm = p.replace("\\", "/")
            if split_token not in norm:
                # Ignore entries from other splits like train2014/test2017
                continue
            # Keep only the base filename (e.g., '000000140556.jpg')
            base = Path(norm).name
            base = self._filepath2imageId(base)
            if base:
                allowed.add(base)
        return allowed

    @staticmethod
    def _filepath2imageId(filepath: str) -> str:
        """
        Convert a filepath to an image ID by removing the .jpg extension and leading zeros.

        Parameters
        ----------
        filepath
            The filepath or filename to convert to an image ID.

        Returns
        -------
        image_id
            The image ID with .jpg extension and leading zeros removed.
            Returns the original filepath if it doesn't end with .jpg.
        """
        if filepath.lower().endswith(".jpg"):
            # Remove the .jpg suffix if present
            filepath = filepath[:-4]
        # Remove leading zeros
        return filepath.lstrip("0")
