# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable

import numpy as np
import torch

from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.bounding_box_processing import batched_nms, box_xywh_to_xyxy
from qai_hub_models.utils.image_processing import resize_pad


def decode(output: list[torch.Tensor], thr: float) -> np.ndarray:
    """
    Decode model output to bounding boxes, class indices and scores.

    Inputs:
        output: list[torch.Tensor]
            Model output.
        thr: float
            Detection threshold. Predictions lower than the thresholds will be discarded.
    Outputs: np.ndarray
        Detection results. Shape is (N, 6). N is the number of detected objects. Each object is
        represented by (class, x1, y1, x2, y2, score)
    """
    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]
    strides = (8, 16, 32)
    result = []
    for s, out in enumerate(output):
        b, h, w, c = out.shape
        out = out.reshape(b, h, w, 3, -1)
        _, ny, nx, na = out.shape[:-1]
        for y in np.arange(ny):
            for x in np.arange(nx):
                for a in np.arange(na):
                    pred = out[0, y, x, a]
                    obj_score = pred[4].sigmoid()
                    cls_score = pred[5:].max().sigmoid()
                    score = obj_score * cls_score
                    if score < thr:
                        continue
                    c = np.argmax(pred[5:])
                    bx = (pred[0].sigmoid() * 2 - 0.5 + x) * strides[s]
                    by = (pred[1].sigmoid() * 2 - 0.5 + y) * strides[s]
                    bw = 4 * pred[2].sigmoid() ** 2 * anchors[s][a][0]
                    bh = 4 * pred[3].sigmoid() ** 2 * anchors[s][a][1]

                    boxes = box_xywh_to_xyxy(
                        torch.from_numpy(np.array([[[bx, by], [bw, bh]]]))
                    )
                    x1 = boxes[0][0][0].round()
                    y1 = boxes[0][0][1].round()
                    x2 = boxes[0][1][0].round()
                    y2 = boxes[0][1][1].round()
                    result.append([c, x1, y1, x2, y2, score])
    return np.array(result, dtype=np.float32)


def postprocess(
    output: list[torch.Tensor],
    scale: float,
    pad: list[int],
    conf_thr: float,
    iou_thr: float,
) -> np.ndarray:
    """
    Post process model output.
    Inputs:
        output: list[torch.Tensor]
            Multi-scale model output.
        scale: float
            Scaling factor from input image and model input.
        pad: list[int]
            Padding sizes from input image and model input.
        conf_thr: float
            Confidence threshold of detections.
        iou_thr: float
            IoU threshold for non maximum suppression.
    Outputs: np.ndarray
        Detected object. Shape is (N, 6). N is the number of detected objects. Each object is
        represented by (class, x1, y1, x2, y2, score)
    """
    result = decode(output, conf_thr)

    result_final = []
    for c in [0, 1]:
        idx = result[:, 0] == c
        boxes, scores = batched_nms(
            iou_thr,
            0,
            torch.from_numpy(result[idx, 1:5]).unsqueeze_(0),
            torch.from_numpy(result[idx, -1]).unsqueeze_(0),
        )
        scores[0].unsqueeze_(-1)
        result_final.append(
            torch.concat([torch.zeros_like(scores[0]) + c, boxes[0], scores[0]], 1)
        )
    result_final_arr = torch.concat(result_final).numpy()
    result_final_arr[:, 1:5] = (
        (result_final_arr[:, 1:5] - np.array([pad[0], pad[1], pad[0], pad[1]])) / scale
    ).round()
    return result_final_arr


class BodyDetectionApp:
    """Body detection application"""

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize BodyDetectionApp.

        Inputs:
            model: Callable[[torch.Tensor], torch.Tensor]
                Detection model.
        """
        self.model = model

    def detect(self, imgfile: str, height: int, width: int, conf: float) -> np.ndarray:
        """
        Detect objects from input images.

        Inputs:
            imgfile: str
                Input image file
            height: int
                Model input height.
            width: int
                Model input width.
            conf: float
                Detection threshold.
        Outputs: np.ndarray
            Detection result. Shape is (N, 6). N is the number of detected objects. Each object is represented by
            (cls_id, x1, y1, x2, y2, score)
        """
        img = np.array(load_image(imgfile))
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0) / 255.0
        input, scale, pad = resize_pad(img, (height, width))
        output = self.model(input)
        for t, o in enumerate(output):
            output[t] = o.permute(0, 2, 3, 1).detach()
        result = postprocess(output, scale, pad, conf, 0.5)
        return result
