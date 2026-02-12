# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from detectron2.modeling import GeneralizedRCNN
from torch.nn import functional as F
from typing_extensions import Self

from qai_hub_models.models._shared.detectron2.model import Detectron2
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.base_model import CollectionModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"


class Detectron2ProposalGenerator(Detectron2):
    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.model = model
        self.pixel_mean = model.pixel_mean
        self.pixel_std = model.pixel_std
        self.backbone = model.backbone
        self.proposal_generator = model.proposal_generator

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image
            Pixel values pre-processed with shape (B, 3, H, W).
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        feature : torch.Tensor
            The "res4" feature map from backbone with shape (B, 1024, H//16, W//16)
        proposal : torch.Tensor
            The proposals for image, with shape (B, num_proposals, 4) in xyxy format.
        objectness_logits : torch.Tensor
            The objectness logits for image, with shape (B, num_proposals,)
        """
        # Detectron2 RCNN:
        # https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/modeling/meta_arch/rcnn.py#L178
        image = (image[:, [2, 1, 0]] - (self.pixel_mean / 255)) / (self.pixel_std / 255)
        feature = self.backbone(image)

        # Detectron2 RPN:
        # https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/modeling/proposal_generator/rpn.py#L431
        features = [feature[f] for f in self.proposal_generator.in_features]
        anchors = self.proposal_generator.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.proposal_generator.rpn_head(
            features
        )
        pred_objectness_logits = [
            score.permute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            x.permute(0, 2, 3, 1).reshape(
                x.shape[0], -1, self.proposal_generator.anchor_generator.box_dim
            )
            for x in pred_anchor_deltas
        ]
        proposals = self.proposal_generator._decode_proposals(
            anchors, pred_anchor_deltas
        )

        return feature["res4"], proposals[0], pred_objectness_logits[0]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 800,
        width: int = 800,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["feature", "proposals", "score"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


class Detectron2ROIHead(Detectron2):
    def __init__(self, model: GeneralizedRCNN) -> None:
        super().__init__()
        self.model = model
        self.roi_heads = model.roi_heads
        self.box_predictor = model.roi_heads.box_predictor

    def forward(
        self, features: torch.Tensor, proposals_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features
            The "res4" feature map from backbone with shape (1, 1024, H//16, W//16).
        proposals_boxes
            The proposals for image, with shape (1, num_proposals, 4)  in xyxy format.

        Returns
        -------
        boxes : torch.Tensor
            A tensor of shape (1, num_proposals, 4)  in xyxy format containing the predicted boxes.
        scores : torch.Tensor
            A tensor of shape (1, num_proposals) containing the scores for each box.
        classes : torch.Tensor
            A tensor of shape (1, num_proposals) containing the labels for each box.
        """
        # Detectron2 ROI heads:
        # https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/modeling/roi_heads/roi_heads.py#L459
        batch_size, num_prop, _ = proposals_boxes.shape
        indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=proposals_boxes.dtype),
            num_prop,
        )
        proposals_boxes = proposals_boxes.reshape(batch_size * num_prop, 4)
        pooler_fmt_boxes = torch.cat([indices[:, None], proposals_boxes], dim=1)

        x = self.roi_heads.pooler.level_poolers[0](features, pooler_fmt_boxes)
        box_features = self.roi_heads.res5(x)

        scores, proposal_deltas = self.box_predictor(box_features.mean(dim=[2, 3]))

        # Detectron2 Fast R-CNN:
        # https://github.com/facebookresearch/detectron2/blob/fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/modeling/roi_heads/fast_rcnn.py#L465
        boxes = self.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas,
            proposals_boxes,
        )
        scores = scores.reshape(batch_size, num_prop, -1)
        if self.box_predictor.use_sigmoid_ce:
            scores = scores.sigmoid()
        else:
            scores = F.softmax(scores, dim=-1)

        scores = scores[..., :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = boxes.view(batch_size, num_prop, num_bbox_reg_classes, 4)

        classes = torch.argmax(scores, dim=-1)
        indices = classes[0]
        boxes = boxes[:, torch.arange(indices.shape[0]), indices]
        scores = scores[:, torch.arange(indices.shape[0]), indices]
        return boxes, scores, classes

    @staticmethod
    def get_input_spec(
        height: int = 50, width: int = 50, num_boxes: int = 200
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "features": ((1, 1024, height, width), "float32"),
            "proposals_boxes": ((1, num_boxes, 4), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "classes"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["features"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Any = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True --truncate_64bit_io True"

        return compile_options


@CollectionModel.add_component(Detectron2ProposalGenerator)
@CollectionModel.add_component(Detectron2ROIHead)
class Detectron2Detection(CollectionModel):
    def __init__(
        self,
        proposal_generator: Detectron2ProposalGenerator,
        roi_head: Detectron2ROIHead,
    ) -> None:
        super().__init__(*[proposal_generator, roi_head])
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head

    @classmethod
    def from_pretrained(cls, config: str = DEFAULT_CONFIG) -> Self:
        return cls(
            Detectron2ProposalGenerator.from_pretrained(config),
            Detectron2ROIHead.from_pretrained(config),
        )
