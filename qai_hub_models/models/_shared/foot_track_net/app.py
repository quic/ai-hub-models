# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt
import torch

from qai_hub_models.utils.bounding_box_processing import get_iou

CLASSNAME_TO_ID_MAP = {"face": 0, "person": 1}


def id_to_classname(id: int) -> str:
    """CLASSNAME_TO_ID_MAP traverse the ID, return the corresponding class name"""
    for k, v in CLASSNAME_TO_ID_MAP.items():
        if v == id:
            return k
    raise RuntimeError(f"Class for id {id} not found.")


def restructure_topk(scores: torch.Tensor, K: int = 20) -> tuple[torch.Tensor, ...]:
    """
    cutomized function for top_k specific for this the FootTrackNet. Wil restructure the original coordinates, class id from the floored index.
    After top k operation. this will specifically decoding the coordinates, class from the topk result.
    parameters:
        scores:  the heatmap scores in flat shape
        K: how many top k to be kept.
    return:
        topk_scores: the scorse list for the top k.
        topk_inds: the index list for the top k.
        topk_clses: the class list for the top k.
        topk_ys: the y coordinate list for the top k.
        topk_xs: the x coordinate list for the top k.
    """
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(
        scores.reshape(batch, -1), min(K, batch * cat * height * width)
    )
    topk_clses = (topk_inds // (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class BBox_landmarks:
    x: float | np.float64
    y: float | np.float64
    r: float | np.float64
    b: float | np.float64

    def __init__(
        self,
        label: str,
        xyrb: list[int] | npt.NDArray[np.int32],
        score: float | int = 0,
        landmark: list | np.ndarray | None = None,
        vis: list | np.ndarray | None = None,
    ):
        """
        A bounding box plus landmarks structure to hold the hierarchical result.
        parameters:
            label:str the class label
            xyrb: 4 array or list for bbox left, top,  right bottom coordinates
            score: the score of the detection
            landmark: 17x2 the landmark of the joints [[x1,y1], [x2,y2]...]
            vis: 17 the visiblity of the joints.
        """
        self.label = label
        self.score = score
        self.landmark = landmark
        self.vis = vis
        self.x, self.y, self.r, self.b = xyrb
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    @property
    def label_prop(self) -> str:
        return self.label

    @label_prop.setter
    def label_prop(self, newvalue: str):
        self.label = newvalue

    @property
    def haslandmark(self) -> bool:
        return self.landmark is not None

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue


def nms_bbox_landmark(
    objs: list[BBox_landmarks], iou: float = 0.5
) -> list[BBox_landmarks]:
    """
    nms function customized to work on the BBox_landmarks objects list.
    parameter:
        objs: the list of the BBox_landmarks objects.
    return:
        the rest of the BBox_landmarks after nms operation.
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
            if (
                flags[j] == 0
                and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
            ):
                flags[j] = 1
    return keep


def drawbbox(
    image: np.ndarray,
    bbox: BBox_landmarks,
    color: list[float] | tuple[float],
    thickness: int = 2,
    landmarkcolor: tuple | list = (0, 0, 255),
    visibility: list | np.ndarray | None = None,
    joint_to_visualize: list = [0, 15, 16],
    visibility_thresh: float = 0.05,
) -> np.ndarray:
    """
    draw a bounding box and landmarks on the input image based on the detection result in BBox_landmarks.
    parameters:
        image: the input image in cv2 format.
        bbox: the detection result in format of BBox_landmarks
        color:the color for the result
        thickness: the thickness of the boundary
        landmarkcolor: the color for the landmark
        visiblity: the visibility of the landmarks.
        joint_to_visualize: which joint to be visualized.
        visibility_thresh: the thresh to deem as the landmark visible or not when drawing it.
    return:
        the image after drawing the result
    """

    x, y, r, b = (int(bb + 0.5) for bb in np.array(bbox.box).astype(int))
    # 3DMM adjustment,  reuse the bbox structure
    if bbox.label_prop == 0:
        cx, cy = (r + x) // 2, (b + y) // 2
        offset = max(r - x, b - y) // 2
        x2 = int(cx - offset)
        y2 = int(cy - offset)
        r2 = int(cx + offset)
        b2 = int(cy + offset)
        cv2.rectangle(image, (x2, y2, r2 - x2 + 1, b2 - y2 + 1), color, thickness, 16)

    else:
        cv2.rectangle(image, (x, y, r - x + 1, b - y + 1), color, thickness, 16)

    if bbox.landmark is not None:
        for i in range(len(bbox.landmark)):
            x, y = bbox.landmark[i][:2]

            if not joint_to_visualize or i not in joint_to_visualize:
                continue
            if visibility is not None and visibility[i] > visibility_thresh:
                cv2.circle(image, (int(x), int(y)), 4, landmarkcolor, -1, 16)
            else:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1, 16)
    return image


def detect_images_multiclass_fb(
    output_hm: torch.Tensor,
    output_tlrb: torch.Tensor,
    output_landmark: torch.Tensor,
    vis: torch.Tensor,
    threshold: list | np.ndarray = [0.7, 0.7, 0.7],
    stride: int = 4,
    n_lmk: int = 17,
) -> list:
    """
    Get the detection result from the model raw output tensors.
    parameters:
        output_hm: N,C,H,W the model heatmap output.
        output_tlrb: N,12,H,W the model bbox output.
        output_landmark: N,34,H,W the model output_landmark output.
        vis: N,17,H,W the model visiblity output
        threshold: 3 the threshold for each class.
        stride: the stride of the output map comparing to input.
        n_lmk: the landmark number.
    return:
        detection result: list[BBox_landmarks]

    """
    _, num_classes, hm_height, hm_width = output_hm.shape
    hm = output_hm[0].reshape(1, num_classes, hm_height, hm_width)
    hm = hm[:, :2]

    tlrb = (
        output_tlrb[0]
        .cpu()
        .data.numpy()
        .reshape(1, num_classes * 4, hm_height, hm_width)
    )

    landmark = output_landmark[0].cpu().data.numpy().reshape(1, -1, hm_height, hm_width)
    vis = vis[0].cpu().data.numpy().reshape(1, -1, hm_height, hm_width)
    nmskey = hm

    kscore, kinds, kcls, kys, kxs = restructure_topk(nmskey, 1000)

    kys = kys.cpu().data.numpy().astype(np.int32)
    kxs = kxs.cpu().data.numpy().astype(np.int32)
    kcls = kcls.cpu().data.numpy().astype(np.int32)
    kscore = kscore.cpu().data.numpy().astype(np.float32)
    kinds = kinds.cpu().data.numpy().astype(np.int32)

    key: list[list[np.int32 | np.float32]] = [
        [],  # [kys..]
        [],  # [kxs..]
        [],  # [score..]
        [],  # [class..]
        [],  # [kinds..]
    ]

    score_fc = []
    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        thr = threshold[kcls[0, ind]]
        if kcls[0, ind] == 0:
            score_fc.append(kscore[0, ind])
        if score > thr:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            key[2].append(score)
            key[3].append(kcls[0, ind])
            key[4].append(kinds[0, ind])

    imboxs = []
    if key[0] is not None and len(key[0]) > 0:
        ky, kx = key[0], key[1]
        classes = key[3]
        scores = key[2]

        for i in range(len(kx)):
            class_ = int(classes[i])
            cx, cy = kx[i], ky[i]
            x1, y1, x2, y2 = tlrb[0, class_ * 4 : (class_ + 1) * 4, cy, cx]
            x1, y1, x2, y2 = (
                np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])
            ) * stride  # back to world

            if class_ == 1:  # face person, only person has landmark otherwise None
                x5y5 = landmark[0, : n_lmk * 2, cy, cx]
                x5y5 = (x5y5 + np.array([cx] * n_lmk + [cy] * n_lmk)) * stride
                boxlandmark = np.array(list(zip(x5y5[:n_lmk], x5y5[n_lmk:])))
                box_vis = vis[0, :, cy, cx].tolist()
            else:
                boxlandmark = None
                box_vis = None
            imboxs.append(
                BBox_landmarks(
                    label=str(class_),
                    xyrb=np.array([x1, y1, x2, y2]),
                    score=scores[i].item(),
                    landmark=boxlandmark,
                    vis=box_vis,
                )
            )
    return imboxs


def postprocess(
    output, threshhold, iou_thr
) -> tuple[list[BBox_landmarks], list[BBox_landmarks]]:
    """
    Get the detection result from the model raw output tensors.
    parameters:
        output: N,C,H,W the model heatmap/bbox/output_landmark/visiblity output.
        threshold: 3 the threshold for each class.
        iou_thr: 3 the iou threshold for each class.
    return:
        face result: list[BBox_landmarks]
        person result: list[BBox_landmarks]
    """
    heatmap = output[0]
    bbox = output[1]
    landmark = output[2]
    landmark_visiblity = output[3]

    stride = 4
    num_landmarks = 17
    objs = detect_images_multiclass_fb(
        heatmap,
        bbox,
        landmark,
        threshold=threshhold,
        stride=stride,
        n_lmk=num_landmarks,
        vis=landmark_visiblity,
    )

    objs_face = []
    objs_person = []

    for obj in objs:
        label = id_to_classname(int(obj.label_prop))
        if label == "face":
            objs_face.append(obj)
        elif label == "person":
            objs_person.append(obj)

    objs_face = nms_bbox_landmark(objs_face, iou=iou_thr[0])
    objs_person = nms_bbox_landmark(objs_person, iou=iou_thr[1])

    return objs_face, objs_person


class FootTrackNet_App:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with DDRNet.

    The app uses 1 model:
        * FootTrackNet

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FootTrackNet inference
        * Convert the output to two lists of BBox_landmarks objects for face and body.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor], if_norm=True):
        """
        model: the input model
        if_norm: if do the image normalization
        """
        self.model = model
        self.threshhold = [0.6, 0.7, 0.7]  # threshold for each detector, 0.6 original
        self.iou_thr = [0.2, 0.5, 0.5]  # iou threshold
        self.if_norm = if_norm

    def predict(self, *args, **kwargs):
        return self.det_image(*args, **kwargs)

    def det_image(
        self, pixel_values: torch.Tensor
    ) -> tuple[list[BBox_landmarks], list[BBox_landmarks]]:
        """
        return two lists,  objs_face, objs_person.
        Each list contains the object of BBox_landmarks which contains the bbox and landmark info. Please refer to BBox definition.

        Parameters:
            pixel_values
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns:
            objs_face: a list of BBox_landmarks for face  list[BBox_landmarks]
            objs_person: a list of BBox_landmarks for person  list[BBox_landmarks]
        """

        output = self.model(pixel_values)

        return postprocess(output, self.threshhold, self.iou_thr)
