# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.yolonas.app import YoloNASDetectionApp
from qai_hub_models.models.yolonas.demo import main as demo_main
from qai_hub_models.models.yolonas.model import MODEL_ASSET_VERSION, MODEL_ID, YoloNAS
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.bounding_box_processing import get_iou
from qai_hub_models.utils.testing import skip_clone_repo_check

GT_BOXES = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolonas_boxes.npy"
)


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    app = YoloNASDetectionApp(YoloNAS.from_pretrained(), nms_score_threshold=0.7)
    boxes = app.predict_boxes_from_image(image, raw_output=True)[0][0].numpy()
    print(boxes.shape)
    boxes_gt = load_numpy(GT_BOXES)
    boxes = sorted(boxes, key=lambda box: box[0])
    boxes_gt = sorted(boxes_gt, key=lambda box: box[0])
    assert len(boxes) == len(boxes_gt)
    ious = [get_iou(box, box_gt) for box, box_gt in zip(boxes, boxes_gt)]
    for iou in ious:
        assert iou > 0.95


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
