# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.utils.bounding_box_processing import get_iou


class BBox:
    # Bounding Box
    def __init__(
        self,
        label: str,
        xyrb: list[int],
        score: float = 0,
        landmark: list | None = None,
        rotate: bool = False,
    ):
        """
        A bounding box plus landmarks structure to hold the hierarchical result.
        parameters:
            label:str the class label
            xyrb: 4 list for bbox left, top,  right bottom coordinates
            score:the score of the deteciton
            landmark: 10x2 the landmark of the joints [[x1,y1], [x2,y2]...]
        """
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate

        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = (
            ",".join([str(item[:2]) for item in self.landmark])
            if self.landmark is not None
            else "empty"
        )
        return (
            f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, "
            + f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"
        )

    @property
    def width(self) -> int:
        return self.r - self.x + 1

    @property
    def height(self) -> int:
        return self.b - self.y + 1

    @property
    def box(self) -> list[int]:
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue: list[int]) -> None:
        self.x, self.y, self.r, self.b = newvalue

    @property
    def haslandmark(self) -> bool:
        return self.landmark is not None

    @property
    def xywh(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def nms(objs: list[BBox], iou: float = 0.5) -> list[BBox]:
    """
    nms function customized to work on the BBox objects list.
    parameter:
        objs: the list of the BBox objects.
    return:
        the rest of the BBox after nms operation.
    """
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            # if flags[j] == 0 and obj.iou(objs[j]) > iou:
            if (
                flags[j] == 0
                and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
            ):
                flags[j] = 1
    return keep


def detect(
    model: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    threshold: float = 0.2,
    nms_iou: float = 0.2,
    stride: int = 8,
) -> list[BBox]:
    hm, box, landmark = model(image)
    hm = hm.sigmoid()
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    lens = ((hm == hm_pool).float() * hm).view(1, -1).cpu().shape[1]
    scores, indices = (
        ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(min(lens, 2000))
    )

    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list(torch.div(indices, hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())

    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[0, :, cy, cx].cpu().data.numpy()
        xyrb: list[int] = (
            (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        ).tolist()
        x5y5 = landmark[0, :, cy, cx].cpu().data.numpy()
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride

        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(BBox("0", xyrb=xyrb, score=score, landmark=box_landmark))

    if nms_iou != -1:
        return nms(objs, iou=nms_iou)
    return objs


class FaceDetLiteApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceDetLite.

    The app uses 1 model:
        * FaceDetLite

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FaceDetLite inference
        * Output list of face Bounding Box objects.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.run_inference_on_image(*args, **kwargs)

    def run_inference_on_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
    ) -> list[list[int | float]]:
        """
        Return the corresponding output by running inference on input image.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            objs_face: a list of BBox for face  list[BBox]
        """
        assert pixel_values_or_image is not None, "pixel_values_or_image is None"
        img = pixel_values_or_image

        if isinstance(img, Image.Image):
            img_array = np.asarray(img)
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            raise RuntimeError("Invalid format")

        img_array = (img_array.astype("float32") / 255.0 - 0.442) / 0.280
        img_array = img_array[np.newaxis, ...]
        img_tensor = torch.Tensor(img_array)
        img_tensor = img_tensor[:, :, :, -1]

        img_tensor = img_tensor[np.newaxis, ...]
        dets = detect(self.model, img_tensor, threshold=0.55, nms_iou=-1, stride=8)
        res = []
        for n in range(0, len(dets)):
            xmin, ymin, w, h = dets[n].xywh
            score = dets[n].score

            L = int(xmin)
            R = int(xmin + w)
            T = int(ymin)
            B = int(ymin + h)
            W = int(w)
            H = int(h)

            if L < 0 or T < 0 or R >= 640 or B >= 480:
                if L < 0:
                    L = 0
                if T < 0:
                    T = 0
                if R >= 640:
                    R = 640 - 1
                if B >= 480:
                    B = 480 - 1

            # Enlarge bounding box to cover more face area
            b_Left = L - int(W * 0.05)
            b_Top = T - int(H * 0.05)
            b_Width = int(W * 1.1)
            b_Height = int(H * 1.1)

            if (
                b_Left >= 0
                and b_Top >= 0
                and b_Width - 1 + b_Left < 640
                and b_Height - 1 + b_Top < 480
            ):
                L = b_Left
                T = b_Top
                W = b_Width
                H = b_Height
                R = W - 1 + L
                B = H - 1 + T

            res.append([L, T, W, H, score])
        return res
