# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import cv2
import torch
from mmpose.apis import MMPoseInferencer

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.printing import suppress_stdout

DEFAULT_INFERENCER_ARCH = "rtmpose-m_8xb64-270e_coco-wholebody-256x192"


class CocoWholeBodyDataset(CocoBodyDataset):
    """
    Wrapper class around CocoWholeBody Human Pose dataset
    http://images.cocodataset.org/
    COCO-WholeBody keypoints::
        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints
        In total, 133 keypoints for wholebody pose estimation.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 256,
        input_width: int = 192,
        num_samples: int = -1,
    ) -> None:
        super().__init__(split, input_height, input_width, num_samples)
        with suppress_stdout():
            self.inference = MMPoseInferencer(
                DEFAULT_INFERENCER_ARCH, device=torch.device(type="cpu")
            )

    def __getitem__(self, idx: int) -> tuple[Any, tuple[Any, Any, Any, Any]]:
        """
        Returns a tuple of input image tensor and label data.
        label data is a List with the following entries:
            - imageId (int): The ID of the image.
            - category_ud (int) : the category ID
            - center (list[float]): The center coordinates of the bounding box.
            - scale (torch.Tensor) : Scaling factor.
        """
        (
            file_name,
            image_id,
            category_id,
            center,
            scale,
        ) = self.kpt_db[idx]
        img_path = self.image_dir / file_name
        data_numpy = cv2.imread(
            str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(data_numpy)
        inputs = self.inference.preprocess(NHWC_int_numpy_frames, batch_size=1)
        proc_inputs, _ = list(inputs)[0]
        proc_inputs_ = proc_inputs["inputs"][0]
        image = proc_inputs_.to(dtype=torch.float32)
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]
        return image, (image_id, category_id, scale, bbox)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 500
