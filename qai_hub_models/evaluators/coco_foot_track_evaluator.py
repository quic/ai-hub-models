# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
import datetime
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import torch

from qai_hub_models.datasets.coco_foot_track_dataset import CocoFootTrackDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models._shared.foot_track_net.app import BBox_landmarks
from qai_hub_models.utils.bounding_box_processing import get_iou
from qai_hub_models.utils.printing import suppress_stdout

CLASSNAME_TO_ID_MAP = {"face": 0, "person": 1}


class CocoFootTrackNetEvaluator(BaseEvaluator):
    """Evaluator for keypoint-based pose estimation using COCO-style mAP."""

    def __init__(self, in_vis_thre=0.2):
        """
        Args:
            coco_gt: COCO ground truth dataset.
        """
        self.predictions = []
        self.in_vis_thre = in_vis_thre
        self.threshhold = [0.5, 0.5, 0.5]  # threshold for each detector, 0.6 original
        self.iou_thr = [0.2, 0.5, 0.5]
        self.coco_gt = CocoFootTrackDataset().cocoGt

    def reset(self):
        """Resets the collected predictions."""
        self.predictions = []

    def undo_resize_pad_BBox(self, bbox: BBox_landmarks, scale: float, padding: list):
        """
        undo the resize and pad in place of the BBox_landmarks object.
        operation in place to replace the inner coordinates
        Parameters:
            scale: single scale from original to target image.
            pad: left, top padding size
        Return:
            None.
        """
        if bbox.haslandmark and bbox.landmark is not None:
            for lmk in bbox.landmark:
                lmk[0] = (lmk[0] - padding[0]) / scale
                lmk[1] = (lmk[1] - padding[1]) / scale
        bbox.x = (bbox.x - padding[0]) / scale
        bbox.y = (bbox.y - padding[1]) / scale
        bbox.r = (bbox.r - padding[0]) / scale
        bbox.b = (bbox.b - padding[1]) / scale

        return

    def add_batch(self, output: torch.Tensor, gt_data: list[torch.Tensor]):
        """
        Collects model predictions in COCO format, handling both single and batched keypoints.

        Args:
            output: Raw model outputs (heatmaps).
            gt_data: Ground truth data from dataset containing (image_id, category_id, center, scale).
        """

        threshold = self.threshhold
        iou_thr = self.iou_thr
        image_ids, category_ids, center, scale = gt_data

        heatmap = output[0]
        bbox = output[1]
        landmark = output[2]
        landmark_visibility = output[3]
        batch_size = heatmap.shape[0]
        stride = 4
        num_landmarks = 17
        for idx in range(batch_size):
            objs = self.detect_images_multiclass_fb(
                heatmap[idx].unsqueeze(0),
                bbox[idx].unsqueeze(0),
                landmark[idx].unsqueeze(0),
                threshold=threshold,
                stride=stride,
                n_lmk=num_landmarks,
                vis=landmark_visibility[idx].unsqueeze(0),
            )
            objs_person = []
            for obj in objs:
                label = self.id_to_classname(int(obj.label_prop))
                if label == "person":
                    objs_person.append(obj)
            objs_person = self.nms_bbox_landmark(objs_person, iou=iou_thr[1])
            for obj in objs_person:
                self.undo_resize_pad_BBox(obj, scale[idx], [0, 0])
                x, y, r, b = (int(bb + 0.5) for bb in np.array(obj.box).astype(int))
                b_box = [x, y, r - x + 1, b - y + 1]
                keypoints = []
                for i in range(len(obj.landmark)):
                    x, y = obj.landmark[i][:2]
                    visibility = obj.vis[i]
                    keypoints.extend(
                        [int(x), int(y), 2 if visibility > self.in_vis_thre else 0]
                    )

                prediction = {
                    "image_id": int(image_ids[idx]),
                    "category_id": int(category_ids[idx]),
                    "bbox": b_box,
                    "keypoints": keypoints,
                    "score": float(obj.score),
                    "center": list(center[idx]),
                    "scale": scale[idx],
                }
                self.predictions.append(prediction)

    def nms_bbox_landmark(
        self, objs: list[BBox_landmarks], iou: float = 0.5
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

    def detect_images_multiclass_fb(
        self,
        output_hm: torch.Tensor,
        output_tlrb: torch.Tensor,
        output_landmark: torch.Tensor | None = None,
        vis: torch.Tensor | None = None,
        threshold: list | np.ndarray = [0.7, 0.7, 0.7],
        stride: int = 4,
        n_lmk: int = 17,
    ) -> list:
        _, num_classes, hm_height, hm_width = output_hm.shape
        hm = output_hm[0].reshape(1, num_classes, hm_height, hm_width)
        hm = hm[:, :2]

        tlrb = (
            output_tlrb[0]
            .cpu()
            .data.numpy()
            .reshape(1, num_classes * 4, hm_height, hm_width)
        )
        if output_landmark is not None:
            landmark = (
                output_landmark[0]
                .cpu()
                .data.numpy()
                .reshape(1, -1, hm_height, hm_width)
            )
        else:
            raise ValueError("output_landmark is None, expected a tensor.")
        if vis is not None:
            vis = vis[0].cpu().data.numpy().reshape(1, -1, hm_height, hm_width)
        else:
            raise ValueError("vis is None, expected a tensor.")
        nmskey = hm

        kscore, kinds, kcls, kys, kxs = self.restructure_topk(nmskey, 1000)
        kys = kys.cpu().data.numpy().astype(np.int32)
        kxs = kxs.cpu().data.numpy().astype(np.int32)
        kcls = kcls.cpu().data.numpy().astype(np.int32)
        kscore = kscore.cpu().data.numpy().astype(np.float32)
        kinds = kinds.cpu().data.numpy().astype(np.int32)

        key: list[list[int]] = [[], [], [], [], []]

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
                class_ = classes[i]
                cx, cy = kx[i], ky[i]
                x1, y1, x2, y2 = tlrb[0, class_ * 4 : (class_ + 1) * 4, cy, cx]
                x1, y1, x2, y2 = (
                    np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])
                ) * stride  # back to world

                if class_ == 1:  # face person, only person has landmakr otherwise None
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
                        score=scores[i],
                        landmark=boxlandmark,
                        vis=box_vis,
                    )
                )
        return imboxs

    def id_to_classname(self, id: int) -> str:
        """CLASSNAME_TO_ID_MAP traverse the ID, return the corresponding class name"""
        for k, v in CLASSNAME_TO_ID_MAP.items():
            if v == id:
                return k
        raise ValueError(f"ID {id} not found in CLASSNAME_TO_ID_MAP")

    def restructure_topk(
        self, scores: torch.Tensor, K: int = 20
    ) -> tuple[Any, Any, Any, Any, Any]:
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

    def get_coco_mAP(self) -> dict[str, Any]:
        """
        Computes COCO-style mAP using COCOfooteval.

        Returns:
            A dictionary with AP values (mAP, AP@0.5, etc.).
        """
        pred_image_ids = [p["image_id"] for p in self.predictions]

        res = copy.deepcopy(self.predictions)
        with suppress_stdout():
            coco_dt = self.coco_gt.loadRes(res)
            coco_eval = COCOfooteval(self.coco_gt, coco_dt, "keypoints")
            coco_eval.params.kpt_oks_sigmas = coco_eval.params.kpt_oks_sigmas[-2:]
            coco_eval.params.catIds = [1]
            coco_eval.params.imgIds = pred_image_ids
            coco_eval.params.useSegm = None
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {"AP": coco_eval.stats[0], "AP@.5": coco_eval.stats[1]}

    def get_accuracy_score(self) -> float:
        """Returns the overall mAP score."""
        return self.get_coco_mAP()["AP"]

    def formatted_accuracy(self) -> str:
        """Formats the mAP score for display."""
        results = self.get_coco_mAP()
        return f"mAP: {results['AP']:.3f}, AP@.5: {results['AP@.5']:.3f}"


class COCOfooteval:
    _paramsEval: Optional[Params] = None
    params: Params
    evalImgs: list

    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        """
        Initialize COCOfooteval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        if not iouType:
            print("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = []  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = None  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = []  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(f"useSegm (deprecated) is not None. Running {p.iouType} evaluation")
        print(f"Evaluate annotation type *{p.iouType}*")
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print(f"DONE (t={toc - tic:0.2f}s).")

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])[15 * 3 : 17 * 3]
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])[15 * 3 : 17 * 3]
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros(k)
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = (
            self.ious[imgId, catId][:, gtind]
            if len(self.ious[imgId, catId]) > 0
            else self.ious[imgId, catId]
        )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape(
            (1, len(dt))
        )
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M)
        )  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        if self._paramsEval is None:
            raise Exception(
                "self._paramsEval is not initialized; please run evaluate() first"
            )
        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA: set[tuple] = {
            tuple(x) if isinstance(x, Iterable) else (x,) for x in _pe.areaRng
        }
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    dtIg = np.concatenate(
                        [e["dtIgnore"][:, 0:maxDet] for e in E], axis=1
                    )[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except IndexError:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        print(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}"
                if iouThr is None
                else f"{iouThr:0.2f}"
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
