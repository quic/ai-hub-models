# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to evaluate accuracy (mAP) of a yolov6 model.
Packages to install: pycocotools, object-detection-metrics==0.4.post1, shapely
"""

from torch.utils.data import DataLoader

from qai_hub_models.datasets.coco import CocoDataset, collate_fn
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models.yolov6.model import YoloV6

if __name__ == "__main__":
    # Load dataset.
    dataset = CocoDataset()
    # Pass it to data loader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=False
    )

    # Load model
    model = YoloV6.from_pretrained()

    # Instantiate the evaluator
    evaluator = DetectionEvaluator(
        image_height=640,
        image_width=640,
        nms_score_threshold=0.3,
        nms_iou_threshold=0.5,
    )

    # Pass batches of data through the model.
    evaluator.add_from_dataset(model, dataloader, eval_iterations=1000)
    print(f"mAP: {evaluator.mAP:.1%}")
