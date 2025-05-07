# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from panopticapi.evaluation import PQStat
from panopticapi.utils import rgb2id

from qai_hub_models.datasets.coco_panoptic_seg import CocoPanopticSegmentationDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.mask2former.app import Mask2FormerApp as app


class PanopticSegmentationEvaluator(BaseEvaluator):
    """Evaluator for panoptic segmentation metrics (PQ, SQ, RQ)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.dataset = CocoPanopticSegmentationDataset()
        self.annotations = self.dataset.annotations
        self.label_divisor = 1000
        self.ignore_label = 0
        self._init_mappings()
        self.reset()

    def _init_mappings(self) -> None:
        self.img_ann_map = {
            ann["image_id"]: ann for ann in self.annotations["annotations"]
        }
        self.categories = {el["id"]: el for el in self.annotations["categories"]}
        sorted_cats = sorted(self.categories.values(), key=lambda x: x["id"])
        self.coco_to_contiguous = {
            cat["id"]: idx for idx, cat in enumerate(sorted_cats)
        }
        self.contiguous_to_coco = {v: k for k, v in self.coco_to_contiguous.items()}
        self.preprocessed_gt_segments = {}
        for img_id, ann in self.img_ann_map.items():
            self.preprocessed_gt_segments[img_id] = [
                {
                    "id": s["id"],
                    "category_id": self.coco_to_contiguous[s["category_id"]],
                    "area": s["area"],
                    "iscrowd": s.get("iscrowd", 0),
                }
                for s in ann.get("segments_info", [])
            ]

    def reset(self) -> None:
        """Reset the evaluation statistics."""
        self.pq_stat = PQStat()

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Process model predictions and ground truth for panoptic segmentation evaluation.

        Args:
            output (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Model predictions with class logits, class labels, and mask logits.
            gt_data (tuple[torch.Tensor, torch.Tensor]): Ground truth panoptic masks and image IDs.
        """
        pred_scores, pred_labels, pred_masks_logits = output
        batch_results = app.post_process_panoptic_segmentation(
            pred_scores, pred_labels, pred_masks_logits
        )
        batched_gt_masks, batched_img_ids = gt_data

        for i in range(batched_gt_masks.shape[0]):
            result = batch_results[i]
            gt_mask = rgb2id(batched_gt_masks[i].cpu().numpy())
            img_id = batched_img_ids[i].item()

            # Process GT
            gt_segments = self.preprocessed_gt_segments.get(img_id, [])

            # Process predictions
            pred_mask = result["segmentation"].cpu().numpy()

            segments_info: list[dict] = []

            for s in result["segments_info"]:
                coco_cat_id = self.contiguous_to_coco.get(s["label_id"])
                if coco_cat_id is None:
                    continue
                instance_id = len(segments_info) + 1
                panoptic_id = coco_cat_id * self.label_divisor + instance_id
                area = np.sum(pred_mask == (s["id"] % self.label_divisor))
                segments_info.append(
                    {
                        "id": panoptic_id,
                        "category_id": s["label_id"],
                        "area": int(area),
                        "iscrowd": 0,
                        "score": s["score"],
                    }
                )

            # Update pred mask with panoptic IDs
            final_pred_mask = np.zeros_like(pred_mask)
            for seg in segments_info:
                instance_id = seg["id"] % self.label_divisor
                final_pred_mask[pred_mask == instance_id] = seg["id"]

            self._evaluate_single_image(
                pred_mask=final_pred_mask,
                gt_mask=gt_mask,
                pred_segments_info=segments_info,
                gt_segments_info=gt_segments,
            )

    def _evaluate_single_image(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        pred_segments_info: list[dict],
        gt_segments_info: list[dict],
    ) -> PQStat:
        """Evaluate a single image's prediction against ground truth.

        Args:
            pred_mask (np.ndarray): Predicted panoptic mask of shape (H, W) with panoptic IDs (category_id * label_divisor + instance_id).
            gt_mask (np.ndarray): Ground truth panoptic mask of shape (H, W) with panoptic IDs or 0 for ignored regions.
            pred_segments_info (list[dict]): Predicted segment info with id, category_id, area, iscrowd (0), score.
            gt_segments_info (list[dict]): Ground truth segment info with id, category_id, area, iscrowd (0 or 1).

        Returns:
            PQStat: Updated panoptic quality statistics for the image.
        """

        pq_stat = self.pq_compute_single_image(
            gt_mask, pred_mask, gt_segments_info, pred_segments_info
        )
        self.pq_stat += pq_stat

    def pq_compute_single_image(
        self,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        gt_segments_info: list[dict],
        pred_segments_info: list[dict],
        ignore_label: int = 0,
    ) -> PQStat:
        """Compute panoptic quality statistics for a single image.

        Args:
            gt_mask (np.ndarray): Ground truth panoptic mask of shape (H, W) with panoptic IDs or 0 for ignored regions.
            pred_mask (np.ndarray): Predicted panoptic mask of shape (H, W) with panoptic IDs (category_id * label_divisor + instance_id).
            gt_segments_info (list[dict]): Ground truth segment info with id, category_id, area, iscrowd (0 or 1).
            pred_segments_info (list[dict]): Predicted segment info with id, category_id, area, iscrowd (0), score.
            ignore_label (int): Value for ignored regions (default: 0).

        Returns:
            PQStat: Statistics with true positives, false positives, false negatives, and IoU for matched segments.
        """
        pq_stat = PQStat()
        VOID = ignore_label
        OFFSET = self.label_divisor * self.label_divisor

        # Use provided segment info directly
        gt_segms = {el["id"]: el for el in gt_segments_info}
        pred_segms = {el["id"]: el for el in pred_segments_info}

        # Update areas from masks
        for seg_id, count in zip(*np.unique(pred_mask, return_counts=True)):
            if seg_id in pred_segms and seg_id != VOID:
                pred_segms[seg_id]["area"] = int(count)
        for seg_id, count in zip(*np.unique(gt_mask, return_counts=True)):
            if seg_id in gt_segms and seg_id != VOID:
                gt_segms[seg_id]["area"] = int(count)

        # Confusion matrix calculation
        pan_gt_pred = gt_mask.astype(np.uint64) * OFFSET + pred_mask.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            if gt_id in gt_segms and pred_id in pred_segms:
                gt_pred_map[(gt_id, pred_id)] = intersection

        # Count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for (gt_id, pred_id), intersection in gt_pred_map.items():
            if gt_id not in gt_segms or pred_id not in pred_segms:
                continue
            if gt_segms[gt_id]["iscrowd"] == 1:
                continue
            # Extract category_id from panoptic_id (using label_divisor)
            gt_cat_id = gt_segms[gt_id]["category_id"]
            pred_cat_id = pred_segms[pred_id]["category_id"]
            if gt_cat_id == pred_cat_id:
                union = (
                    gt_segms[gt_id]["area"]
                    + pred_segms[pred_id]["area"]
                    - intersection
                    - gt_pred_map.get((VOID, pred_id), 0)
                )
                iou = intersection / union if union > 0 else 0
                if iou > 0.5:
                    pq_stat[gt_cat_id].tp += 1
                    pq_stat[gt_cat_id].iou += iou
                    gt_matched.add(gt_id)
                    pred_matched.add(pred_id)

        # Count false negatives
        for gt_id, gt_info in gt_segms.items():
            if gt_id in gt_matched or gt_info["iscrowd"] == 1 or gt_id == VOID:
                continue
            pq_stat[gt_info["category_id"]].fn += 1

        # Count false positives
        crowd_labels_dict = {
            gt_info["category_id"]: gt_id
            for gt_id, gt_info in gt_segms.items()
            if gt_info["iscrowd"] == 1
        }
        for pred_id, pred_info in pred_segms.items():
            if pred_id in pred_matched or pred_id == VOID:
                continue
            intersection = gt_pred_map.get((VOID, pred_id), 0)
            if pred_info["category_id"] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info["category_id"]], pred_id), 0
                )
            if intersection / pred_info["area"] <= 0.5:
                pq_stat[pred_info["category_id"]].fp += 1

        return pq_stat

    def compute_pq(self) -> dict[str, dict]:
        """Compute panoptic quality metrics for all, things, and stuff categories.

        Returns:
            dict[str, dict]: Metrics for "All", "Things", "Stuff" with pq, sq, rq, and n (number of categories).
        """
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        return {
            name: self.pq_stat.pq_average(self.categories, isthing=isthing)[0]
            for name, isthing in metrics
        }

    def get_accuracy_score(self) -> float:
        """Return the PQ score for all categories."""
        return self.compute_pq()["All"]["pq"]

    def formatted_accuracy(self) -> str:
        """Return formatted PQ score as a percentage."""
        return f"{self.get_accuracy_score() * 100:.1f} PQ"
