# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataloader import default_collate

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    setup_fiftyone_env,
)
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    transform_resize_pad_normalized_coordinates,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


class CocoDatasetClass(Enum):
    ALL_CLASSES = 91
    SUBSET_CLASSES = 80


DATASET_ID = "coco"
DATASET_ASSET_VERSION = 1


def collate_fn(batch):
    try:
        image, gt = batch[0][0], batch[0][1]
        image_id, height, width, boxes, labels = gt
        new_list = []
        new_list.append(default_collate([i for i in image if torch.is_tensor(i)]))
        target = (
            torch.tensor(image_id),
            torch.tensor(height),
            torch.tensor(width),
            default_collate([i for i in boxes if torch.is_tensor(i)]),
            default_collate([i for i in labels if torch.is_tensor(i)]),
        )
        new_list.append(target)
        return new_list
    except Exception:
        return [], ([], [], [], [], [], [])


class CocoDataset(BaseDataset):
    """
    Wrapper class around COCO dataset https://cocodataset.org/

    Contains object detection samples and labels spanning 80 or 91 classes.

    This wrapper supports the train and val splits of the 2017 version.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        max_boxes: int = 100,
        num_samples: int = 5000,
        num_classes: CocoDatasetClass = CocoDatasetClass.SUBSET_CLASSES,
        label_types: list[Literal["detections", "segmentations"]] | None = None,
    ):
        """
        Parameters
        ----------
        split
            Whether to use the train or val split of the dataset.

        input_spec
            Model input spec; determines shapes for model input produced by this dataset.

        max_boxes
            The maximum number of boxes for a given sample. Used so that
            when loading multiple samples in a batch via a dataloader, this will
            be the tensor dimension.

            If a sample has fewer than this many boxes, the tensor of boxes
            will be zero padded up to this amount.

            If a sample has more than this many boxes, an exception is thrown.

        num_samples
            Number of data samples to download. Needs to be specified
            during initialization because only as many samples as requested
            are downloaded.

        num_classes
            Number of classes this model detects.

        label_types
            Type of predictions this model produces.
        """
        if label_types is None:
            label_types = ["detections"]
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.label_types = label_types

        # FiftyOne package manages dataset so pass a dummy name for data path
        BaseDataset.__init__(self, "non_existent_dir", split, input_spec)

        # coco labels 91 reference from https://huggingface.co/facebook/detr-resnet-50/blob/main/config.json
        # the mapping is to insert unused and extend to 91 labels.
        label_file = (
            "coco_labels_91.txt"
            if self.num_classes == CocoDatasetClass.ALL_CLASSES
            else "coco_labels.txt"
        )

        counter = 0
        self.label_map = {}
        with open(QAIHM_PACKAGE_ROOT / "labels" / label_file) as f:
            for line in f:
                self.label_map[line.strip()] = counter
                counter += 1

        # input_spec is (h, w) and target_image_size is (w, h)
        input_spec = input_spec or {"image": ((1, 3, 640, 640), "")}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.max_boxes = max_boxes

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Get an item in this dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.


        Returns
        -------
        image
            Input image resized for the network. RGB, floating point range [0-1].

        ground_truth
            image_id
                Image ID within the original dataset
            target_height
                Returns image height.
            target_width
                Returned image width.
            bboxes
                bounding box data with shape (self.max_boxes, 4)
                The 4 are (x1, y1, x2, y2), and are normalized [0, 1] fp values.
                Box coordinates are normalized to reflect their locations in the input image
                    (which has been resized / padded to fit the network input shape).
            labels
                Ground truth predicted category IDs with shape (self.max_boxes)
            num_boxes
                Number of boxes present in the ground truth data. Shape [1].
        """
        from fiftyone.core.sample import SampleView

        # Open PIL image sample.
        sample: SampleView = self.dataset[index : index + 1].first()
        image = Image.open(sample.filepath).convert("RGB")
        src_image_w, src_image_h = image.size

        # Convert to torch (NCHW, range [0, 1]) tensor.
        torch_image = app_to_net_image_inputs(image)[1]
        # Scale and center-pad image to user-requested target image shape.
        scaled_padded_torch_image, scale_factor, pad = resize_pad(
            torch_image, (self.target_h, self.target_w)
        )

        boxes_list = []
        labels_list = []
        if sample.ground_truth is not None:
            for annotation in sample.ground_truth.detections:
                x, y, w, h = annotation.bounding_box
                coords = torch.Tensor([[x, y], [x + w, y + h]])
                transformed_coords = transform_resize_pad_normalized_coordinates(
                    coords,
                    (src_image_w, src_image_h),
                    scaled_padded_torch_image.shape[2:4][::-1],
                    scale_factor,
                    pad,
                )
                boxes_list.append(list(transformed_coords.flatten()))

                # Convert string label to int idx
                labels_list.append(self.label_map[annotation.label])
        boxes = torch.tensor(boxes_list)
        labels = torch.tensor(labels_list)

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4))
            labels = torch.zeros(self.max_boxes)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

        return scaled_padded_torch_image.squeeze(0), (
            int(Path(sample.filepath).name[:-4]),
            self.target_h,
            self.target_w,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _validate_data(self) -> bool:
        return hasattr(self, "dataset")

    def _download_data(self) -> None:
        setup_fiftyone_env()

        # This is an expensive import, so don't want to unnecessarily import it in
        # other files that import datasets/__init__.py
        import fiftyone.zoo as foz

        # Sorting by filepath ensures a deterministic ordering every time this is called
        split_str = "validation" if self.split == DatasetSplit.VAL else "train"
        self.dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split_str,
            label_types=self.label_types,
            max_samples=self.num_samples,
            shuffle=False,
        ).sort_by("filepath")

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 300

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 split",
        )
