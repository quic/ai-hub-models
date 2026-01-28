# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.centerpoint.model import (
    CENTERPOINT_SOURCE_PATCHES,
    COMMIT_HASH,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SOURCE_REPO,
)
from qai_hub_models.utils.asset_loaders import (
    SourceAsRoot,
)
from qai_hub_models.utils.bounding_box_processing_3d import compute_iou_3d_rotated

# *** Imports required for the Data processing ***
with SourceAsRoot(
    SOURCE_REPO,
    COMMIT_HASH,
    MODEL_ID,
    MODEL_ASSET_VERSION,
    source_repo_patches=CENTERPOINT_SOURCE_PATCHES,
):
    from det3d.core import box_torch_ops
    from det3d.datasets.pipelines.formating import Reformat
    from det3d.datasets.pipelines.loading import LoadPointCloudFromFile
    from det3d.datasets.pipelines.preprocess import Preprocess, Voxelization
    from det3d.models.bbox_heads.center_head import _circle_nms
    from tools.demo_utils import visual

from collections.abc import Mapping, Sequence
from typing import Any

from addict import Dict

from qai_hub_models.utils.inference import OnDeviceModel


def convert_numpy_to_tensor(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays within a nested data structure to PyTorch tensors.

    This function traverses the input object, which can be a NumPy array, a mapping
    (e.g., dict), a sequence (e.g., list, tuple), or any combination of these, and
    converts all `numpy.ndarray` instances to `torch.Tensor` using `torch.from_numpy`.

    Parameters
    ----------
    obj
        The input object to be converted. It can be:
        - `numpy.ndarray`
        - A mapping (e.g., dict, OrderedDict) with values that may include NumPy arrays or nested structures
        - A sequence (e.g., list, tuple) containing elements that may include NumPy arrays or nested structures
        - Any other type, which will be returned unchanged

    Returns
    -------
    Any
        A new object with the same structure as `obj`, but with all `numpy.ndarray`
        instances replaced by `torch.Tensor`. Other types are returned as-is.
    """
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(torch.float32)
    if isinstance(obj, Mapping):
        return {k: convert_numpy_to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return [convert_numpy_to_tensor(item) for item in obj]
    return obj


def rotate_nms_pcdet(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    pre_maxsize: int | None = None,
    post_max_size: int | None = None,
) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) on rotated 3D bounding boxes using a simplified IoU calculation.

    This function filters overlapping bounding boxes based on their confidence scores and a specified IoU threshold.
    Rotation is considered in the box format but may be ignored in the IoU computation depending on implementation.

    Parameters
    ----------
    boxes
        Tensor of shape (N, 7) representing 3D bounding boxes in the format:
        [x, y, z, l, w, h, theta]
        - x, y, z : Center coordinates
        - l, w, h : Dimensions of the box
        - theta : Rotation angle (in radians)
    scores
        Tensor of shape (N,) representing confidence scores for each box.
    thresh
        IoU threshold for suppressing overlapping boxes.
    pre_maxsize
        If set, limits the number of boxes considered before applying NMS.
    post_max_size
        If set, limits the number of boxes returned after NMS.

    Returns
    -------
    selected
        Tensor containing the indices of selected boxes after NMS.
    """
    boxes_np = boxes.clone().cpu().numpy()
    scores_np = scores.clone().cpu().numpy()

    # Transform to [x, y, z, w, l, h, -theta - pi/2]
    boxes_np = boxes_np[:, [0, 1, 2, 4, 3, 5, 6]]
    boxes_np[:, -1] = -boxes_np[:, -1] - np.pi / 2

    # Sort scores in descending order
    order = scores_np.argsort()[::-1].copy()

    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    boxes_np = boxes_np[order]

    keep: list[int] = []

    for i in range(len(boxes_np)):
        current = boxes_np[i]
        should_keep = True
        for j in keep:
            iou = compute_iou_3d_rotated(current, boxes_np[j])
            if iou > thresh:
                should_keep = False
                break
        if should_keep:
            keep.append(i)

    selected = torch.tensor(order[keep], dtype=torch.long)

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected


class CenterPointApp:
    """
    CenterPointApp performs end-to-end inference and visualization using the CenterPoint 3D object detection model.

    This class handles:
        - Preprocessing of LiDAR binary files into model-ready tensors.
        - Running inference using a CenterPoint-compatible model.
        - Visualizing detection results alongside ground truth annotations.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | OnDeviceModel,
        cfg: Any,
    ) -> None:
        """
        Initialize the CenterPointApp instance.

        Parameters
        ----------
        model
            The inference model used for 3D object detection. It should accept three
            torch.Tensor inputs and return a tuple of three torch.Tensor outputs.
        cfg
            Configuration object containing test settings and model tasks.

        Attributes
        ----------
        model
            The provided inference model.
        test_cfg
            Test-time configuration extracted from cfg.test_cfg.
        num_classes
            Number of classes for each detection task.
        """
        self.model = model
        self.test_cfg = cfg.test_cfg
        tasks = cfg.model.bbox_head.tasks
        num_classes = [len(t["class_names"]) for t in tasks]
        self.num_classes = num_classes
        box_torch_ops.rotate_nms_pcdet = rotate_nms_pcdet
        box_torch_ops.compute_iou_3d_rotated = compute_iou_3d_rotated

    def preprocess_bin_file(self, bin_path: str) -> dict[str, torch.Tensor]:
        """
        Preprocesses a binary LiDAR file and returns a dictionary of tensors ready for inference.

        Parameters
        ----------
        bin_path
            Path to the binary LiDAR point cloud file.

        Returns
        -------
        dict
            A dictionary containing preprocessed tensors including voxels, coordinates, and number of points.
        """
        info = {
            "lidar_path": bin_path,
            "sweeps": [
                {
                    "lidar_path": bin_path,
                    "transform_matrix": None,
                    "time_lag": 0,
                }
            ],
        }

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": 2,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": "",
                "num_point_features": 5,
                "token": "inference",
            },
            "calib": None,
            "cam": {},
            "mode": "test",
            "virtual": False,
        }

        load = LoadPointCloudFromFile(dataset="NuScenesDataset")
        preprocess = Preprocess(cfg=Dict({"mode": "val", "shuffle_points": False}))
        voxelize = Voxelization(
            cfg=Dict(
                {
                    "range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    "voxel_size": [0.2, 0.2, 8],
                    "max_points_in_voxel": 20,
                    "max_voxel_num": [30000, 60000],
                }
            )
        )
        reformat = Reformat(mode="test")

        for step in [load, preprocess, voxelize, reformat]:
            res, info = step(res, info)

        res_dict: dict[str, torch.Tensor] = convert_numpy_to_tensor(res)
        coor: torch.Tensor = res_dict["coordinates"]
        zeros = torch.zeros(coor.shape[0], 1, dtype=torch.float32)
        res_dict["coordinates"] = torch.cat([zeros, coor], dim=1)

        return res_dict

    def predict(self, lidar_input: dict[str, torch.Tensor]) -> Image.Image:
        """
        Runs inference on the preprocessed input using the CenterPoint model.

        Parameters
        ----------
        lidar_input
            Dictionary containing tensors for 'voxels', 'coordinates', and 'num_points'.
            For input details checkout the model forward section.

        Returns
        -------
        Image
            Rendered PIL image with detections overlaid on the point cloud.
        """
        voxels, coordinates, num_points, lidar_points = (
            lidar_input["voxels"],
            lidar_input["coordinates"],
            lidar_input["num_points"],
            lidar_input["points"],
        )
        batch_box_preds, batch_hm = self.model(voxels, coordinates, num_points)
        return self.visualizer(lidar_points, batch_box_preds, batch_hm)

    def post_processing(
        self,
        batch_box_preds: torch.Tensor,
        batch_hm: torch.Tensor,
        test_cfg: Any,
        post_center_range: torch.Tensor,
        task_id: int,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Apply score/distance filtering and NMS to model outputs for a batch.

        Parameters
        ----------
        batch_box_preds
            Dense 3D bounding box predictions for each BEV grid cell.
            Tensor of shape (B, HxW, 9), where:
                batch_box_preds[..., 0:2]
                    Decoded BEV center coordinates (x, y) in meters.

                batch_box_preds[..., 2]
                    Height of the box center (z) in meters.

                batch_box_preds[..., 3:6]
                    3D box dimensions (width, length, height) in meters.

                batch_box_preds[..., 6:7]
                    Yaw rotation angle in radians.

                batch_box_preds[..., 7:9] (optional)
                    Velocity components (vx, vy) in m/s, if enabled.

        batch_hm
            Center heatmap confidence scores for each BEV grid cell.
            Tensor of shape (B, HxW, C).
            Range: [0, 1]
        test_cfg
            Test-time configuration containing thresholds and NMS settings
        post_center_range
            Tensor of shape (6,) specifying [x_min, y_min, z_min, x_max, y_max, z_max]
            used to filter boxes by center location.
        task_id
            Index of the task head (used for task-specific NMS params).

        Returns
        -------
        prediction_dicts
            A list of prediction dicts for the batch, one per sample, with keys:
            - "box3d_lidar"
                Selected 3D boxes after NMS and filtering.
            - "scores"
                Scores for the selected boxes.
            - "label_preds"
                Predicted class indices for the selected boxes.
        """
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) & (
                box_preds[..., :3] <= post_center_range[3:]
            ).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get("circular_nms", False):
                centers = boxes_for_nms[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(
                    boxes,
                    min_radius=test_cfg.min_radius[task_id],
                    post_max_size=test_cfg.nms.nms_post_max_size,
                )
            else:
                selected = box_torch_ops.rotate_nms_pcdet(
                    boxes_for_nms.float(),
                    scores.float(),
                    thresh=test_cfg.nms.nms_iou_threshold,
                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size,
                )

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                "box3d_lidar": selected_boxes,
                "scores": selected_scores,
                "label_preds": selected_labels,
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts

    def visualizer(
        self,
        points: torch.Tensor,
        batch_box_preds: torch.Tensor,
        batch_hm: torch.Tensor,
    ) -> Image.Image:
        """
        Visualizes the 3D object detection results on the LiDAR point cloud and returns the rendered image.

        This method overlays predicted bounding boxes on the point cloud and returns a PIL image.

        Parameters
        ----------
        points
            Tensor containing the LiDAR point cloud data.
        batch_box_preds
            Dense 3D bounding box predictions for each BEV grid cell.
        batch_hm
            Center heatmap confidence scores for each BEV grid cell.

        Returns
        -------
        image
            The rendered image as a PIL Image object, or None if the image could not be loaded.
        """
        post_center_range = self.test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=torch.float32,
                device="cpu",
            )
        rets = []
        rets.append(
            self.post_processing(
                batch_box_preds, batch_hm, self.test_cfg, post_center_range, 0
            )
        )
        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])
        assert num_samples == 1
        if num_samples == 1:
            ret_list = rets[0]
        else:
            for i in range(num_samples):
                ret = {}
                for k in rets[0][i]:
                    if k in ["box3d_lidar", "scores"]:
                        ret[k] = torch.cat([ret[i][k] for ret in rets])
                    elif k in ["label_preds"]:
                        flag = 0
                        for j, num_class in enumerate(self.num_classes):
                            rets[j][i][k] += flag
                            flag += num_class
                        ret[k] = torch.cat([ret[i][k] for ret in rets])

                ret_list.append(ret)

        return visual(points, ret_list[0], 0)
