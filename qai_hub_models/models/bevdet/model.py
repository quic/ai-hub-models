# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from qai_hub_models.evaluators.nuscenes_evaluator import (
    NuscenesObjectDetectionEvaluator,
)
from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.models.bevdet.model_patches import (
    CenterHead,
    LSSViewTransformerOptimized,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

with patch_mmdet_no_build_deps():
    from mmdet.models.task_modules import BaseBBoxCoder
    from mmdet.registry import MODELS

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# These assets are preprocessed inputs for the model,
# which are source from the nuscene dataset correspond to
# 'scene-0103' with sample_token '3e8750f331d7499e9b5123e9eb70f2e2'
IMAGES = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "imgs.pt")
INV_INTRINS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inv_intrins.pt"
)
SENSOR2KEYEGOS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sensor2keyegos.pt"
)
INV_POST_ROTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inv_post_rots.pt"
)
POST_TRANS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "post_trans.pt"
)

BEVDET_SOURCE_REPOSITORY = "https://github.com/HuangJunJie2017/BEVDet.git"
BEVDET_SOURCE_REPO_COMMIT = "26144be7c11c2972a8930d6ddd6471b8ea900d13"

# Checkpoint is sourced from
# https://github.com/HuangJunJie2017/BEVDet/tree/dev3.0?tab=readme-ov-file#:~:text=30.7-,baidu,-baidu
BEVDET_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "bevdet-r50.pth"
)
"""
Changes in patch file include;
    - To make functional on AI_HUB
        - Changed dependencies to work with newer MMCV / MMDET
"""
BEVDET_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches/bevdet_patch.diff")
    )
]


class BEVDet(BaseModel):
    """
    BEVDet 3D Object Detection

    Parameters
    ----------
        img_backbone (BaseModule):Image encoder backbone.
        img_neck(BaseModule): Image encoder neck.
        img_view_transformer (BaseModule): Image view transformer.
        img_bev_encoder_backbone (BaseModule):  BEV encoder backbone.
        img_bev_encoder_neck (BaseModule):  BEV encoder neck.
        pts_bbox_head (BaseModule): 3D object detection head.
        bboxcoder (BaseBBoxCoder): BBox coder.
    """

    def __init__(
        self,
        img_backbone: BaseModule,
        img_neck: BaseModule,
        img_view_transformer: BaseModule,
        img_bev_encoder_backbone: BaseModule,
        img_bev_encoder_neck: BaseModule,
        pts_bbox_head: BaseModule,
        bbox_coder: BaseBBoxCoder,
    ) -> None:
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.img_view_transformer = img_view_transformer
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck
        self.pts_bbox_head = pts_bbox_head
        self.bboxcoder = bbox_coder

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> BEVDet:
        ckpt = str(BEVDET_CKPT.fetch()) if ckpt is None else ckpt
        with SourceAsRoot(
            BEVDET_SOURCE_REPOSITORY,
            BEVDET_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=BEVDET_SOURCE_PATCHES,
        ):
            cfg = Config.fromfile("./configs/bevdet/bevdet-r50.py")
            cfg.model.train_cfg = None

            def load_module(class_name: str, path: str) -> type:
                module_spec = spec_from_file_location(class_name, location=Path(path))
                if module_spec is None or module_spec.loader is None:
                    raise ValueError(f"Module spec not found for path {path}")

                module = module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                return getattr(module, class_name)

            # Since source repository doesn't support the latest mmcv version
            # loading the classes directly from the file
            CustomFPN = load_module("CustomFPN", "mmdet3d/models/necks/fpn.py")
            FPN_LSS = load_module("FPN_LSS", "mmdet3d/models/necks/lss_fpn.py")
            CustomResNet = load_module(
                "CustomResNet", "mmdet3d/models/backbones/resnet.py"
            )
            CenterPointBBoxCoder = load_module(
                "CenterPointBBoxCoder",
                "mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py",
            )

            cfg.model.pts_bbox_head.bbox_coder.pop("type")
            cfg.model.img_neck.pop("type")
            cfg.model.img_bev_encoder_backbone.pop("type")
            cfg.model.img_bev_encoder_neck.pop("type")

            img_backbone = MODELS.build(cfg.model.img_backbone)
            img_neck = CustomFPN(**cfg.model.img_neck)
            img_view_transformer = LSSViewTransformerOptimized(
                **cfg.model.img_view_transformer
            )
            img_bev_encoder_backbone = CustomResNet(
                **cfg.model.img_bev_encoder_backbone
            )
            img_bev_encoder_neck = FPN_LSS(**cfg.model.img_bev_encoder_neck)
            pts_test_cfg = cfg.model.test_cfg["pts"] if cfg.model.test_cfg else None
            cfg.model.pts_bbox_head.update(test_cfg=pts_test_cfg)
            pts_bbox_head = CenterHead(**cfg.model.pts_bbox_head)
            bbox_coder = CenterPointBBoxCoder(**cfg.model.pts_bbox_head.bbox_coder)

            model = BEVDet(
                img_backbone,
                img_neck,
                img_view_transformer,
                img_bev_encoder_backbone,
                img_bev_encoder_neck,
                pts_bbox_head,
                bbox_coder,
            )

            load_checkpoint(model, ckpt, map_location="cpu")
            model.eval()
        return model

    def forward(
        self,
        image: torch.Tensor,
        sensor2keyegos: torch.Tensor,
        inv_intrins: torch.Tensor,
        inv_post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run BEVDet model and return the 3D bboxes, scores, labels

        Parameters
        ----------
            B = batch size, S = number of cameras, C = 3, H = img height, W = img width
            image: torch.Tensor of shape [B,S*C,H,W] as float32
                Preprocessed image with range[0-1] in RGB format.
            sensor2keyegos: torch.Tensor of shape [B,S,4,4] as float32
                transformation matix to convert from camera sensor
                to ego-vehicle at front camera coordinate frame
            inv_intrins: torch.Tensor of shape [B,S,3,3] as float32
                Inverse of Camera intrinsic matrix
                used to project 2D image coordinates to 3D points
            inv_post_rots: torch.tensor with shape [B, N, 3, 3] as float32
                inverse post rotation matrix in camera coordinate system
            post_trans: torch.tensor with shape [B, N, 1, 3] as float32
                post translation tensor in camera coordinate system

        Returns
        -------
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

        """
        mean = torch.tensor([123.675, 116.28, 103.53]) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375]) / 255.0
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        _, NC, H, W = image.shape
        image = image.view(NC // 3, 3, H, W)[:, [2, 1, 0], ...]
        image = (image - mean) / std

        x = self.img_backbone(image)
        x = self.img_neck(x)
        x = self.img_view_transformer(
            [x, sensor2keyegos, inv_intrins, inv_post_rots, post_trans]
        )
        x = self.img_bev_encoder_backbone(x)
        img_feats = self.img_bev_encoder_neck(x)
        ret_dicts = self.pts_bbox_head(img_feats)

        reg, height, dim, rot, vel, heatmap = (
            ret_dicts[0]["reg"],
            ret_dicts[0]["height"],
            ret_dicts[0]["dim"],
            ret_dicts[0]["rot"],
            ret_dicts[0]["vel"],
            ret_dicts[0]["heatmap"].sigmoid(),
        )

        return reg, height, dim, rot, vel, heatmap

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_cam: int = 6,
        height: int = 256,
        width: int = 704,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "image": (
                (batch_size, num_cam * 3, height, width),
                "float32",
            ),
            "sensor2keyegos": ((batch_size, num_cam, 4, 4), "float32"),
            "inv_intrins": ((batch_size, num_cam, 3, 3), "float32"),
            "inv_post_rots": ((batch_size, num_cam, 3, 3), "float32"),
            "post_trans": ((batch_size, num_cam, 1, 3), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["reg", "height", "dim", "rot", "vel", "heatmap"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_torch(IMAGES)
        B, N, C, H, W = image.shape
        image = image.view(B, N * C, H, W).numpy()
        sensor2keyegos = load_torch(SENSOR2KEYEGOS).numpy()
        inv_intrins = load_torch(INV_INTRINS).numpy()
        inv_post_rots = load_torch(INV_POST_ROTS).numpy()
        post_trans = load_torch(POST_TRANS).numpy()
        return {
            "image": [image],
            "sensor2keyegos": [sensor2keyegos],
            "inv_intrins": [inv_intrins],
            "inv_post_rots": [inv_post_rots],
            "post_trans": [post_trans],
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device=None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"

        return compile_options

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if "--compute_unit" not in profile_options:
            profile_options = (
                profile_options + " --compute_unit cpu"
            )  # Accuracy not regained on NPU
        return profile_options

    def get_evaluator(self) -> NuscenesObjectDetectionEvaluator:
        return NuscenesObjectDetectionEvaluator(self.bboxcoder)

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["nuscenes"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "nuscenes"

    @staticmethod
    def get_hub_litemp_percentage(_) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 30
