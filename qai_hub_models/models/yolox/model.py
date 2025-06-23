# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import nn

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess_split_input
from qai_hub_models.models.yolov3.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov3_demo_640.jpg"
)

SOURCE_REPOSITORY = "https://github.com/Megvii-BaseDetection/YOLOX"
SOURCE_REPO_COMMIT = "d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolox_s.pth"
MODEL_ASSET_VERSION = 1


class YoloX(Yolo):
    """Exportable yolox-m bounding box detector, end-to-end."""

    def __init__(
        self,
        yolox_source_model: torch.nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            from yolox.models.network_blocks import SiLU
            from yolox.models.yolo_head import YOLOXHead
            from yolox.utils import meshgrid, replace_module

        assert isinstance(
            yolox_source_model.head, YOLOXHead
        ), "Only the YOLOXHead defined in yolo_head.py is supported."
        # pytorch SiLU activation function has to be replaced by a custom implementation
        yolox_source_model = replace_module(yolox_source_model, nn.SiLU, SiLU)

        super().__init__()
        self.model = yolox_source_model
        self.yolox_meshgrid = meshgrid
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ):
        """Load yolox-m from a weightfile created by the source Yolox repository."""
        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, weights_name
        ).fetch()
        # Load PyTorch model from disk
        yolox_model = _load_yolox_source_model_from_weights(
            str(checkpoint_path), weights_name
        )
        return cls(yolox_model, include_postprocessing, split_output)

    def yolox_head_forward_with_split_outputs(
        self, xin
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified version of the YOLOX Detector Head's forward() function.
        (See https://github.com/Megvii-BaseDetection/YOLOX/blob/d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a/yolox/models/yolo_head.py#L142)

        This keeps xywh and scores tensors separate, rather than concatenating them into a single tensor.
        Keeping those tensors separate makes quantization viable.
        """

        #  list (one element per anchor set) of:
        #     tuple[
        #          xywh: [b, 4, # anchors, # anchors]
        #              Bounding box anchor logits that are used to compute box xywh.
        #
        #          probs: [b, 81, # anchors, #anchors]
        #              Predicted Probability [0 - 1]
        #               [b, 0, ...] -> Probability of whether the box has an object.
        #               [b, 1:81, ...] -> Probability of which class the object belongs to.
        #     ]
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(
                self.model.head.cls_convs,
                self.model.head.reg_convs,
                self.model.head.strides,
                xin,
            )
        ):
            x = self.model.head.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.model.head.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.model.head.reg_preds[k](reg_feat)
            obj_output = self.model.head.obj_preds[k](reg_feat)

            # [b, 4, # anchors, # anchors]
            xywh = reg_output
            # [b, 1, # anchors, # anchors]
            frame_has_object_probs = obj_output.sigmoid()
            # [b, 80, # anchors, # anchors]
            object_class_probs = cls_output.sigmoid()

            outputs.append(
                (xywh, torch.cat((frame_has_object_probs, object_class_probs), dim=1))
            )

        return self.yolox_head_decode_with_split_outputs(outputs, dtype=xin[0].type())

    def yolox_head_decode_with_split_outputs(
        self, outputs: list[tuple[torch.Tensor, torch.Tensor]], dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modified version of the YOLOX Detector Head's decode_outputs() function.
        (See https://github.com/Megvii-BaseDetection/YOLOX/blob/d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a/yolox/models/yolo_head.py#L235C9-L235C23)

        This keeps xywh and scores tensors separate, rather than concatenating them into a single tensor.
        Keeping those tensors separate makes quantization viable.
        """
        grids = []
        strides = []

        hw = [single_output[0].shape[-2:] for single_output in outputs]
        for (hsize, wsize), stride in zip(hw, self.model.head.strides):
            yv, xv = self.yolox_meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # convert from list of tensors to [b, n_anchors_all, 4]
        raw_xywh = torch.cat(
            [x[0].flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        # convert from list of tensors to [b, n_anchors_all, 81]
        scores = torch.cat([x[1].flatten(start_dim=2) for x in outputs], dim=2).permute(
            0, 2, 1
        )

        return (
            (raw_xywh[..., 0:2] + grids) * strides,  # xy
            torch.exp(raw_xywh[..., 2:4]) * strides,  # wh
            scores,
        )

    def forward(self, image):
        """
        Run YoloX on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float
                   3-channel Color Space: RGB [0-1]

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape [batch, num preds, 4] where 4 == (topleft_x, topleft_y, bottomright_x, bottomright_y)
                scores: torch.Tensor
                    Confidence score that the given box is the predicted class: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            else if self.split_output:
                output_xy: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [x_center, y_center] (box_coordinates)

                output_wh: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [width, height] (box_size)

                output_scores: torch.Tensor
                    Shape is [batch, num_preds, j]
                        where j[0] is [confidence that there is an object in the box]
                        and j[1:81] is [confidence that the detected object is each class]

            else:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, 85]
                        where 85 is structured as follows:
                            [0:4] -> [x_center, y_center, w, h] box_coordinates
                            [5] -> confidence there is an object in the box (1)
                            [6:85] -> confidence that the detected object is each class (80 -- the number of classes)
        """
        # Scale the image pixel values from [0, 1] to [0, 255] as per yolox requirement
        image = image * 255

        # Run model
        fpn_outs = self.model.backbone(image)
        xy, wh, scores = self.yolox_head_forward_with_split_outputs(fpn_outs)

        # Postprocessing
        if self.include_postprocessing:
            boxes, scores, class_idx = detect_postprocess_split_input(xy, wh, scores)
            return boxes, scores, class_idx.to(torch.uint8)

        if self.split_output:
            return xy, wh, scores

        return torch.cat([xy, wh, scores], dim=0)

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True, split_output: bool = False
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        if split_output:
            return ["boxes_xy", "boxes_wh", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing, self.split_output
        )


def _load_yolox_source_model_from_weights(
    weights_path: str, weights_name: str
) -> torch.nn.Module:
    # Load Yolox model from the source repository using the given weights.
    # Returns <source repository>.models.yolo.Model
    with SourceAsRoot(
        SOURCE_REPOSITORY,
        SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        from yolox.exp import get_exp

        weights_name = weights_name.replace("_", "-").replace(".pth", "")
        exp = get_exp(exp_name=weights_name)
        model = exp.get_model()
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to("cpu").eval()

        return model
