# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from qai_hub_models.models.deepbox.model import (
    DEEPBOX_SOURCE_REPO_COMMIT,
    DEEPBOX_SOURCE_REPOSITORY,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    VGG3DDetection,
    Yolo2DDetection,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo
from qai_hub_models.utils.bounding_box_processing import batched_nms
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

with SourceAsRoot(
    DEEPBOX_SOURCE_REPOSITORY,
    DEEPBOX_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
) as repo_path:
    find_replace_in_repo(
        repo_path,
        "library/Math.py",
        "A = np.zeros([4,3], dtype=np.float)",
        "A = np.zeros([4,3], dtype=float)",
    )

    from library.Math import calc_location
    from library.Plotting import plot_3d_box
    from torch_lib import ClassAverages, Dataset


class DeepBoxApp:
    """
    This class consists of "app code" that is required to perform end to end inference with MediaPipe.

    The app uses 2 models:
        * Yolo2DDetection
        * VGG3DDetection

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1]).
        * Detect the object using Yolo2DDetection.
        * For Every Detected Object, Makes the 2D detection to 3D.
        * Map the 3D Bounding boxes to the original input frame.
    """

    def __init__(
        self,
        bbox2D_dectector: Yolo2DDetection,
        bbox3D_dectector: VGG3DDetection,
        nms_score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
    ):
        """
        Construct a DeepBox 3D object detection application.

        Inputs:
            bbox2D_dectector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                The 2D boundary box dectection model.
                Input is an image [N C H W], channel layout is BGR, output is [pred_boxes, pred_scores, pred_class_idx].

            bbox3D_dectector: Callable[[torch.Tensor], tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list]]],
                The 3D boundary box dectection model.
                Input is an image [N C H W], channel layout is BGR, output is [proj_matrix, orient, dim, location].

            nms_score_threshold: float
                Score threshold for when NMS is run on the detector output boxes.

            nms_iou_threshold: float
                IOU threshold for when NMS is run on the detector output boxes.
        """
        self.yolo = bbox2D_dectector
        self.vgg = bbox3D_dectector
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def predict(self, *args, **kwargs):
        # See predict_3d_boxes_from_image.
        return self.detect_image(*args, **kwargs)

    def detect_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> tuple[
        list[npt.NDArray[np.float64]],
        list[np.float64],
        list[npt.NDArray[np.float32]],
        list[list[np.float64]],
    ] | Image.Image:
        """
        From the provided image or tensor, predict the 3d bounding boxes & classes of objects detected within.

        Parameters:
            image: PIL image

        Returns:
            if raw_output is False, returns
                image_with_3d_bounding_boxes: list[PIL.Image]
                    Input image with predicted 3D Bounding Boxes applied
            otherwise, return
                proj_matrixes: list[np.array]
                    camera to img matrix
                orients: list[np.array]
                    global orientations
                dims: list[np.array]
                    dimensions for the 3d bboxes
                locations: list[list]
                    centers of 3d_bboxes
        """

        # Input Prep
        numpy_image = np.array(image)
        (H, W) = numpy_image.shape[:2]
        (H_resized, W_resized) = self.yolo.get_input_spec()["image"][0][-2:]
        image_resized = image.resize((W_resized, H_resized))

        raw_pred_boxes, pred_scores, pred_class_idx = self.detect_2d_bboxes(
            image_resized
        )

        # Converting output floating point box coordinates to the input image's coordinate space
        height_scale = H / H_resized
        width_scale = W / W_resized
        pred_boxes = raw_pred_boxes[0]
        pred_boxes[:, (0, 2)] = pred_boxes[:, (0, 2)] * width_scale
        pred_boxes[:, (1, 3)] = pred_boxes[:, (1, 3)] * height_scale

        # Detect 3d bboxes for each objects detected
        proj_matrixes: list[npt.NDArray[np.float64]] = []
        orients: list[np.float64] = []
        dims: list[npt.NDArray[np.float32]] = []
        locations: list[list[np.float64]] = []
        for i in range(pred_scores[0].shape[0]):
            output = self.detect_3d_bboxes(
                numpy_image, pred_boxes[i], pred_class_idx[0][i]
            )
            if output is None:
                continue
            proj_matrixes.append(output[0])
            orients.append(output[1])
            dims.append(output[2])
            locations.append(output[3])

        if raw_output:
            return proj_matrixes, orients, dims, locations

        img = Image.fromarray(numpy_image)
        return img

    def detect_2d_bboxes(
        self, image_resized: Image.Image
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(image_resized)

        pred_boxes, pred_scores, pred_class_idx = self.yolo(NCHW_fp32_torch_frames)

        pred_boxes, pred_scores, pred_class_idx = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
        )

        return pred_boxes, pred_scores, pred_class_idx

    def detect_3d_bboxes(
        self,
        numpy_image: np.ndarray,
        pred_boxes: torch.Tensor,
        pred_class_idx: torch.Tensor,
    ) -> tuple[
        npt.NDArray[np.float64], np.float64, npt.NDArray[np.float32], list[np.float64]
    ] | None:
        averages = ClassAverages.ClassAverages()
        angle_bins = Dataset.generate_bins(2)

        # Gets the labels and camera calib
        labels_path = os.path.sep.join([repo_path + "/weights", "coco.names"])
        labels = open(labels_path).read().split("\n")
        calib_file = repo_path + "/camera_cal/calib_cam_to_cam.txt"

        x1, y1, x2, y2 = pred_boxes

        # skip invalid bboxes
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return None

        # change the bbox from xyxy tensor to list([xy][xy]) and
        # assign the label for the class
        box_2d = [[int(x1), int(y1)], [int(x2), int(y2)]]
        detected_class = labels[int(pred_class_idx)]
        if detected_class == "person":
            detected_class = "pedestrian"

        # detects only for car, truck, van, tram, cyclist and pedestrian
        if not averages.recognized_class(detected_class):
            return None

        # convert from BGR to RGB
        image_bgr = numpy_image[..., ::-1]
        detectedObject = Dataset.DetectedObject(
            image_bgr, detected_class, box_2d, calib_file
        )

        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix

        # detect the 3d bbox
        [orient, conf, dim] = self.vgg(input_img.unsqueeze(0))
        orient = orient.numpy()[0, :, :]
        conf = conf.numpy()[0, :]
        dim = dim.numpy()[0, :]

        # add avgerage dim of the detected class
        dim += averages.get_item(detected_class)

        # global orientation
        argmax = np.argmax(conf)
        cos, sin = orient[argmax, :]
        alpha = np.arctan2(sin, cos) + angle_bins[argmax] - np.pi
        orient = alpha + theta_ray

        # calculate best_loc, [left_constraints, right_constraints]
        location, X = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)

        # plots 3d boxes
        plot_3d_box(numpy_image, proj_matrix, orient, dim, location)
        return (proj_matrix, orient, dim, location)
