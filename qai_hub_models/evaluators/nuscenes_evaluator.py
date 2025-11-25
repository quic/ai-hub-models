# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps

with patch_mmdet_no_build_deps():
    from mmdet.models.task_modules import BaseBBoxCoder
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from qai_hub_models.datasets.nuscenes import NuscenesDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing_3d import circle_nms


class NuScenesAttribute(Enum):
    """Attributes for NuScenes instances."""

    # moving
    VEHICLE_MOVING = "vehicle.moving"
    CYCLE_WITH_RIDER = "cycle.with_rider"
    PEDESTRIAN_MOVING = "pedestrian.moving"

    # not moving
    VEHICLE_PARKED = "vehicle.parked"
    VEHICLE_STOPPED = "vehicle.stopped"
    CYCLE_WITHOUT_RIDER = "cycle.without_rider"
    PEDESTRIAN_STANDING = "pedestrian.standing"
    NO_ATTRIBUTE = ""


@dataclass
class NuscenesAnnotation:
    """
    Represents a single annotation for an object in the NuScenes dataset.

    Attributes
    ----------
        sample_token (str): Unique identifier for the sample (image/frame) this
                            annotation belongs to.
        translation (list[float]): Bounding box center coordinates [x, y, z] in meters.
        size (list[float]): Bounding box dimensions [width, length, height] in meters.
        rotation (list[float]): Quaternion rotation [w, x, y, z].
        velocity (list[float]): Object velocity [vx, vy, vz] in meters/second.
        detection_name (str): The class name of the detected object (e.g., 'car', 'pedestrian').
        detection_score (float): Confidence score of the detection.
        attribute_name (str): Additional attribute associated with the object (e.g., 'cycle.with_rider').
    """

    sample_token: str
    translation: list[float]
    size: list[float]
    rotation: list[float]
    velocity: list[float]
    detection_name: str
    detection_score: float
    attribute_name: str


class NuscenesObjectDetectionEvaluator(BaseEvaluator):
    """Object Detection Evaluator for Nuscenes dataset."""

    def __init__(
        self,
        bbox_coder: BaseBBoxCoder,
        nms_threshold: float = 4.0,
        nms_post_max_size: int = 500,
        movement_threshold: float = 0.2,
    ):
        self.reset()
        self.bbox_coder = bbox_coder
        self.nusc = NuscenesDataset().nusc
        self.nms_threshold = nms_threshold
        self.nms_post_max_size = nms_post_max_size
        self.movement_threshold = movement_threshold
        self.MovingAttribute = {
            "car": NuScenesAttribute.VEHICLE_MOVING,
            "truck": NuScenesAttribute.VEHICLE_MOVING,
            "construction_vehicle": NuScenesAttribute.VEHICLE_MOVING,
            "bus": NuScenesAttribute.VEHICLE_MOVING,
            "trailer": NuScenesAttribute.VEHICLE_MOVING,
            "barrier": NuScenesAttribute.NO_ATTRIBUTE,
            "motorcycle": NuScenesAttribute.CYCLE_WITH_RIDER,
            "bicycle": NuScenesAttribute.CYCLE_WITH_RIDER,
            "pedestrian": NuScenesAttribute.PEDESTRIAN_MOVING,
            "traffic_cone": NuScenesAttribute.NO_ATTRIBUTE,
        }
        self.NotMovingAttribute = {
            "car": NuScenesAttribute.VEHICLE_PARKED,
            "truck": NuScenesAttribute.VEHICLE_PARKED,
            "construction_vehicle": NuScenesAttribute.VEHICLE_PARKED,
            "bus": NuScenesAttribute.VEHICLE_STOPPED,
            "trailer": NuScenesAttribute.VEHICLE_PARKED,
            "barrier": NuScenesAttribute.NO_ATTRIBUTE,
            "motorcycle": NuScenesAttribute.CYCLE_WITHOUT_RIDER,
            "bicycle": NuScenesAttribute.CYCLE_WITHOUT_RIDER,
            "pedestrian": NuScenesAttribute.PEDESTRIAN_STANDING,
            "traffic_cone": NuScenesAttribute.NO_ATTRIBUTE,
        }

    def reset(self):
        self.nusc_annos = {}

    def add_batch(
        self,
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        gt: tuple[str, torch.Tensor, torch.Tensor],
    ):
        """
        Parameters
        ----------
            output: A tuple of tensors containing the predicted bounding boxes, scores, and class indices.
                reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
            gt: A tuple of tensors containing
                samplet_token (str): Unique identifier for the sample.
                trans (torch.Tensor): ego2global Translation with the
                    shape of [B, 3].
                rots (torch.Tensor): ego2global Rotation with the
                    shape of [B, 4].
        """
        sample_token, trans, rots = gt
        reg, height, dim, rot, vel, heatmap = output

        result_list = self.get_bboxes(reg, height, dim, rot, vel, heatmap)
        for i in range(len(sample_token)):
            boxes_pt, scores_pt, labels_pt = result_list[i]
            boxes = boxes_pt.numpy()
            scores = scores_pt.numpy()
            labels = labels_pt.int()

            translate = trans[i].tolist()
            rotate = Quaternion(rots[i].tolist())
            annos = []
            for j, box in enumerate(boxes):
                name = list(self.MovingAttribute.keys())[labels[j]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rotate)
                nusc_box.translate(translate)

                if (
                    np.sqrt(nusc_box.velocity[0] ** 2 + nusc_box.velocity[1] ** 2)
                    > self.movement_threshold
                ):
                    attr = self.MovingAttribute[name]
                else:
                    attr = self.NotMovingAttribute[name]
                nusc_anno = NuscenesAnnotation(
                    sample_token=sample_token[i],
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=float(scores[j]),
                    attribute_name=attr.value,
                )
                annos.append(nusc_anno.__dict__)
            # other cams results of the same frame should be concatenated
            if sample_token[i] in self.nusc_annos:
                self.nusc_annos[sample_token[i]].extend(annos)
            else:
                self.nusc_annos[sample_token[i]] = annos

    def get_bboxes(
        self,
        reg: torch.Tensor,
        height: torch.Tensor,
        dim: torch.Tensor,
        rot: torch.Tensor,
        vel: torch.Tensor,
        heatmap: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate bboxes from bbox head predictions.

        Parameters
        ----------
            reg (torch.Tensor): 2D regression value with the
                shape of [B, 2, H, W].
            height (torch.Tensor): Height value with the
                shape of [B, 1, H, W].
            dim (torch.Tensor): Size value with the shape
                of [B, 3, H, W].
            rot (torch.Tensor): Rotation value with the
                shape of [B, 2, H, W].
            vel (torch.Tensor): Velocity value with the
                shape of [B, 2, H, W].
            heatmap (torch.Tensor): Heatmap with the shape of
                [B, N, H, W].

        Returns
        -------
            ret list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                Decoded bbox with shape (Num_pred, 9):
                    (x, y, z, w, l, h, yaw, vx, vy),
                    where x, y, z, w, l, h are in meters,
                    yaw in radian and vx, vy in m/s.
                scores with shape (Num_pred,) and
                labels with shape (Num_pred,) after nms.
        """
        # Decode bboxes from the given inputs
        # https://github.com/HuangJunJie2017/BEVDet/blob/26144be7c11c2972a8930d6ddd6471b8ea900d13/mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py#L117
        decoded_outputs = self.bbox_coder.decode(
            heatmap,
            rot[:, 0].unsqueeze(1),
            rot[:, 1].unsqueeze(1),
            height,
            torch.exp(dim),
            vel,
            reg=reg,
            task_id=0,
        )

        ret = []
        for i in range(len(decoded_outputs)):
            boxes3d = decoded_outputs[i]["bboxes"]
            scores = decoded_outputs[i]["scores"]
            labels = decoded_outputs[i]["labels"]
            centers = boxes3d[:, [0, 1]]
            boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
            selected_indices = circle_nms(
                boxes.detach().numpy(),
                thresh=self.nms_threshold,
                post_max_size=self.nms_post_max_size,
            )

            bboxes = boxes3d[selected_indices]
            scores = scores[selected_indices]
            labels = labels[selected_indices]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            ret.append((bboxes, scores, labels))
        return ret

    def evaluate(self) -> tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.

        return: A tuple of high-level and the raw metric data.
        """
        cfg = config_factory("detection_cvpr_2019")
        max_boxes_per_sample = cfg.max_boxes_per_sample

        # Deserialize results and get meta data.
        pred_boxes = EvalBoxes.deserialize(self.nusc_annos, DetectionBox)

        # Check that each sample has no more than x predicted boxes.
        for sample_token in pred_boxes.sample_tokens:
            assert len(pred_boxes.boxes[sample_token]) <= max_boxes_per_sample, (
                f"Error: Only {max_boxes_per_sample} boxes per sample allowed!"
            )

        all_gt_boxes = load_gt(self.nusc, "mini_val", DetectionBox, verbose=False)

        pred_sample_tokens = pred_boxes.sample_tokens

        gt_boxes = EvalBoxes()
        for sample_token in pred_sample_tokens:
            boxes = all_gt_boxes[sample_token].copy()
            gt_boxes.add_boxes(sample_token, boxes)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), (
            "Samples in split doesn't match samples in predictions."
        )

        # Calculates and adds the distance of each box's center from the ego vehicle.
        pred_boxes = add_center_dist(self.nusc, pred_boxes)
        gt_boxes = add_center_dist(self.nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        pred_boxes = filter_eval_boxes(
            self.nusc, pred_boxes, cfg.class_range, verbose=False
        )
        gt_boxes = filter_eval_boxes(
            self.nusc, gt_boxes, cfg.class_range, verbose=False
        )

        # Accumulate metric data for all classes and distance thresholds.
        metric_data_list = DetectionMetricDataList()
        for class_name in cfg.class_names:
            for dist_th in cfg.dist_ths:
                md = accumulate(
                    gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th
                )
                metric_data_list.set(class_name, dist_th, md)

        # Calculate metrics from the data.
        metrics = DetectionMetrics(cfg)
        for class_name in cfg.class_names:
            # Compute APs.
            for dist_th in cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
                # Barrier and traffic_cone are static objects with no meaningful motion attributes:
                # - attr_err: Not applicable (only vehicles have motion attributes like "moving/parked")
                # - vel_err: Not applicable (static objects have zero velocity by definition)
                # - orient_err: Not applicable for traffic_cone (symmetric shape makes orientation undefined)
                # Per nuScenes evaluation protocol, these metrics are set to NaN for static classes
                if (
                    class_name in ["traffic_cone"]
                    and metric_name
                    in [
                        "attr_err",
                        "vel_err",
                        "orient_err",
                    ]
                ) or (
                    class_name in ["barrier"]
                    and metric_name
                    in [
                        "attr_err",
                        "vel_err",
                    ]
                ):
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        return metrics.mean_ap, metrics.nd_score

    def get_accuracy_score(self) -> float:
        mAP, _NDS = self.evaluate()
        return mAP

    def formatted_accuracy(self) -> str:
        mAP, NDS = self.evaluate()
        return f"{NDS:.4f} NDS, {mAP:.4f} mAP"
