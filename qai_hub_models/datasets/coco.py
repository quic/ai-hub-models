# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataloader import default_collate

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit, setup_fiftyone_env
from qai_hub_models.models._shared.yolo.model import DEFAULT_YOLO_IMAGE_INPUT_HW, Yolo
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
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
        label_types: list[str] = ["detections"],
    ):
        """
        Parameters:
            target_image_size: The size to which the input images will be resized.
            split: Whether to use the train or val split of the dataset.
            max_boxes: The maximum number of boxes for a given sample. Used so that
                when loading multiple samples in a batch via a dataloader, this will
                be the tensor dimension.

                If a sample has fewer than this many boxes, the tensor of boxes
                will be zero padded up to this amount.

                If a sample has more than this many boxes, an exception is thrown.
            num_samples: Number of data samples to download. Needs to be specified
                during initialization because only as many samples as requested
                are downloaded.
            use_all_classes : Flag to indicate whether to use 91 classes (DETR) or 80 classes (YOLO).
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.label_types = label_types

        # FiftyOne package manages dataset so pass a dummy name for data path
        BaseDataset.__init__(self, "non_existent_dir", split)

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
            for line in f.readlines():
                self.label_map[line.strip()] = counter
                counter += 1

        input_spec = input_spec or Yolo.get_input_spec()
        # input_spec is (h, w) and target_image_size is (w, h)
        self.target_image_size = (input_spec["image"][0][3], input_spec["image"][0][2])
        self.x_ratio = self.target_image_size[0] / DEFAULT_YOLO_IMAGE_INPUT_HW
        self.y_ratio = self.target_image_size[1] / DEFAULT_YOLO_IMAGE_INPUT_HW
        self.max_boxes = max_boxes

    def __getitem__(self, item):
        """
        Returns a tuple of input image tensor and label data.

        Label data is a tuple with the following entries:
          - Image ID within the original dataset
          - height (in pixels)
          - width (in pixels)
          - bounding box data with shape (self.max_boxes, 4)
            - The 4 should be normalized (x, y, w, h)
          - labels with shape (self.max_boxes,)
          - number of actual boxes present
        """
        from fiftyone.core.sample import SampleView

        sample = self.dataset[item : item + 1].first()
        assert isinstance(sample, SampleView)
        image = Image.open(sample.filepath).convert("RGB")
        width, height = image.size
        boxes = []
        labels = []
        if sample.ground_truth is not None:
            for annotation in sample.ground_truth.detections:
                if annotation.label not in self.label_map:
                    print(f"Warning: Invalid label {annotation.label}")
                    continue
                x, y, w, h = annotation.bounding_box
                boxes.append(
                    [
                        x * self.x_ratio,
                        y * self.y_ratio,
                        (x + w) * self.x_ratio,
                        (y + h) * self.y_ratio,
                    ]
                )
                # Convert string label to int idx
                labels.append(self.label_map[annotation.label])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((100, 4))
            labels = torch.zeros(100)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

        image = image.resize(self.target_image_size)
        image = app_to_net_image_inputs(image)[1].squeeze(0)
        return image, (
            int(Path(sample.filepath).name[:-4]),
            height,
            width,
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
        """
        The default value for how many samples to run in each inference job.
        """
        return 300
