# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import torch
from mmengine.config import Config
from mmengine.model import BaseModule
from qai_hub.client import Device
from torchpack.utils.config import configs

from qai_hub_models.extern.mmdet import patch_mmdet_no_build_deps
from qai_hub_models.models.bevfusion_det.model_patch import (
    PatchMerging_forward_optimized,
    bev_pool,
    patched_centerhead_get_task_detections,
    patched_get_cam_feats,
    patched_lss_forward,
    patched_topk,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.bounding_box_processing_3d import (
    circle_nms as patched_circle_nms,
)
from qai_hub_models.utils.bounding_box_processing_3d import onnx_atan2
from qai_hub_models.utils.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.optimization import optimized_cumsum
from qai_hub_models.utils.window_partitioning import (
    WindowMSA_forward_optimized,
    window_partition_optimized,
    window_reverse_optimized,
)

with patch_mmdet_no_build_deps():
    from mmdet.models.backbones.swin import ShiftWindowMSA, WindowMSA
    from mmdet.models.layers import PatchMerging
    from mmdet.registry import MODELS

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 8

# checkpoint from  https://github.com/mit-han-lab/bevfusion/blob/main/README.md#:~:text=41.21-,Link,-LiDAR%2DOnly%20Baseline
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "camera-only-det.pth"
)
BEVF_SOURCE_REPOSITORY = "https://github.com/mit-han-lab/bevfusion.git"
BEVF_SOURCE_REPO_COMMIT = "326653dc06e0938edf1aae7d01efcd158ba83de5"

BEVF_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches/bevfusion_patches.diff")
    )
]


def _apply_optimizations() -> None:
    """Apply optimized methods for WindowMSA, ShiftWindowMSA, and PatchMerging."""
    WindowMSA.forward = WindowMSA_forward_optimized
    ShiftWindowMSA.window_partition = window_partition_optimized
    ShiftWindowMSA.window_reverse = window_reverse_optimized
    PatchMerging.forward = PatchMerging_forward_optimized


class BEVFusionEncoder1(BaseModel):
    def __init__(self, backbone: BaseModule, neck: BaseModule) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    @classmethod
    def from_pretrained(
        cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())
    ) -> BEVFusionEncoder1:
        _apply_optimizations()
        return BEVFusion.from_pretrained(ckpt).encoder1

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input images using precomputed mean and std, reshapes them for processing,
        passes them through the backbone (Swin Transformer) and neck (GeneralizedLSSFPN),
        and returns the first feature map.

        Parameters
        ----------
            imgs (torch.Tensor): Input tensor of shape (batch_size, 18, height, width) and range of [0-1],
            where 18 = 6 cameras * 3 channels (RGB).

        Returns
        -------
            torch.Tensor: Feature tensor of shape (batch_size, 6, 256, 32, 88).
        """
        B, NC, H, W = imgs.size()
        imgs_reshaped = imgs.reshape(B * (NC // 3), 3, H, W)
        imgs_normalized = normalize_image_torchvision(imgs_reshaped)
        x = self.backbone(imgs_normalized)
        x = self.neck(x)
        return x[0]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = 256, width: int = 704
    ) -> InputSpec:
        return {"imgs": ((batch_size, 6 * 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["feature_map"]


class BEVFusionEncoder2(BaseModel):
    def __init__(self, vtransform: BaseModule) -> None:
        super().__init__()
        self.vtransform = vtransform

    @classmethod
    def from_pretrained(
        cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())
    ) -> BEVFusionEncoder2:
        _apply_optimizations()
        return BEVFusion.from_pretrained(ckpt).encoder2

    def forward(
        self,
        x: torch.Tensor,
        intrins: torch.Tensor,
        camera2lidars: torch.Tensor,
        inv_post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects camera features into BEV space and applies pooling.

        Parameters
        ----------
            x (torch.Tensor): Feature tensor of shape (batch_size, 6, 256, 32, 88).
            intrins (torch.Tensor): Camera intrinsics of shape (batch_size, 6, 3, 3).
            camera2lidars (torch.Tensor): Camera-to-LiDAR transformations of shape (batch_size, 6, 4, 4).
            inv_post_rots (torch.Tensor): Inverse rotation matrices of shape (batch_size, 6, 3, 3).
            post_trans (torch.Tensor): Post-transformation translations of shape (batch_size, 6, 1, 3).

        Returns
        -------
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x (torch.Tensor): Pooled features of shape (1993728, 80).
                - lengths (torch.Tensor): Lengths tensor of shape (59000,).
                - geom_feats (torch.Tensor): Geometric features of shape (2, 59000).
        """
        x, geom_feats, ranks = self.vtransform.forward(
            x, intrins, camera2lidars, inv_post_rots, post_trans
        )
        x, lengths, geom_feats = bev_pool(x, geom_feats, ranks)
        return x, lengths, geom_feats

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = 256, width: int = 704
    ) -> InputSpec:
        return {
            "img": ((batch_size, 6, 256, 32, 88), "float32"),
            "intrins": ((batch_size, 6, 3, 3), "float32"),
            "camera2lidars": ((batch_size, 6, 4, 4), "float32"),
            "inv_post_rots": ((batch_size, 6, 3, 3), "float32"),
            "post_trans": ((batch_size, 6, 1, 3), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["pooled_features", "lengths", "geom_feats"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True --truncate_64bit_io True"
        return compile_options


class BEVFusionEncoder3(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def from_pretrained(
        cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())
    ) -> BEVFusionEncoder3:
        return BEVFusion.from_pretrained(ckpt).encoder3

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Processes lengths tensor with cumulative sum and aggregates features using cumulative sums
        and indexing based on lengths.

        Parameters
        ----------
            x (torch.Tensor): Input features of shape (batch_size, 1993728, 80).
            lengths (torch.Tensor): Input lengths tensor of shape (batch_size, 59000).

        Returns
        -------
            - x (torch.Tensor): Aggregated features of shape (59000, 80).
        """
        lengths = torch.cumsum(lengths, dim=0).long()

        x_reshaped = x.view(96, 118, 176, 80).float()
        x = optimized_cumsum(x_reshaped).view(x.shape)
        x = x.reshape(-1, 80)
        return x[lengths, :]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = 256, width: int = 704
    ) -> InputSpec:
        return {
            "img": ((batch_size, 1993728, 80), "float32"),
            "length": ((batch_size, 59000), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["aggregated_features"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True --truncate_64bit_io True"
        return compile_options


class BEVFusionEncoder4(BaseModel):
    def __init__(self, vtransform: BaseModule) -> None:
        super().__init__()
        self.vtransform = vtransform

    @classmethod
    def from_pretrained(
        cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())
    ) -> BEVFusionEncoder4:
        return BEVFusion.from_pretrained(ckpt).encoder4

    def forward(
        self, segment_sums: torch.Tensor, geom_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes differences in segment sums, maps them to a BEV grid using geometric indices,
        and downsamples the result to produce a structured BEV feature map.

        Parameters
        ----------
            segment_sums (torch.Tensor): Aggregated features of shape (1, 59000, 80).
            geom_feats (torch.Tensor): Geometric indices of shape (1, 2, 59000).

        Returns
        -------
            torch.Tensor: BEV grid features of shape (1, 80, 256, 256).
        """
        segment_sums = segment_sums.reshape(-1, 80)
        n = segment_sums.size(0)
        segment_sums[1:] = segment_sums[1:] - segment_sums[: n - 1]
        x_pos_vals = geom_feats[0, 0]
        channel_vals = geom_feats[0, 1]
        out = torch.zeros((1, 256, 256, 80))
        ba = torch.zeros_like(x_pos_vals)
        out.index_put_(
            (ba.long(), x_pos_vals.long(), channel_vals.long()), segment_sums
        )
        x = out.permute(0, 3, 1, 2).contiguous()
        return self.vtransform.downsample(x)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = 256, width: int = 704
    ) -> InputSpec:
        return {
            "segment": ((1, 59000, 80), "float32"),
            "geom": ((1, 2, 59000), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["bev_grid"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"
        return compile_options


class BEVFusionDecoder(BaseModel):
    def __init__(
        self, backbone: BaseModule, neck: BaseModule, heads: BaseModule
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads

    @classmethod
    def from_pretrained(
        cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())
    ) -> BEVFusionDecoder:
        return BEVFusion.from_pretrained(ckpt).decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes BEV features through a backbone (GeneralizedResNet), neck (LSSFPN),
        and heads (CenterHead) to generate 3D bounding box predictions, scores, and labels.

        Parameters
        ----------
            x (torch.Tensor): BEV feature tensor of shape (batch_size, 80, 128, 128).

        Returns
        -------
            torch.Tensor: Concatenated tensor containing all task-specific outputs
                      (regression, height, dimension, rotation, velocity, heatmap)
                      with shape (batch_size, 70, 128, 128).
        """
        x = self.backbone(x)
        x = self.neck(x)
        ret_dicts = self.heads(x)

        # Pack tensors into a single tensor for torch.jit.trace compatibility
        tensors = []
        for task_dict in ret_dicts:
            task_dict = task_dict[0]
            for key in ["reg", "height", "dim", "rot", "vel", "heatmap"]:
                tensor = task_dict[key]
                tensors.append(tensor)

        return torch.cat(tensors, dim=1)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = 128, width: int = 128
    ) -> InputSpec:
        return {"x": ((batch_size, 80, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["outputs"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True"
        return compile_options


@CollectionModel.add_component(BEVFusionEncoder1)
@CollectionModel.add_component(BEVFusionEncoder2)
@CollectionModel.add_component(BEVFusionEncoder3)
@CollectionModel.add_component(BEVFusionEncoder4)
@CollectionModel.add_component(BEVFusionDecoder)
class BEVFusion(CollectionModel):
    def __init__(
        self,
        encoder1: BEVFusionEncoder1,
        encoder2: BEVFusionEncoder2,
        encoder3: BEVFusionEncoder3,
        encoder4: BEVFusionEncoder4,
        decoder: BEVFusionDecoder,
    ) -> None:
        super().__init__(encoder1, encoder2, encoder3, encoder4, decoder)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.encoder4 = encoder4
        self.decoder = decoder

    @classmethod
    def from_pretrained(cls, ckpt: str = str(DEFAULT_WEIGHTS.fetch())) -> BEVFusion:
        _apply_optimizations()
        with SourceAsRoot(
            BEVF_SOURCE_REPOSITORY,
            BEVF_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=BEVF_SOURCE_PATCHES,
        ) as repo_path:

            def load_module(class_name: str, path: str) -> Any:
                module_spec = spec_from_file_location(class_name, location=Path(path))
                if module_spec is None or module_spec.loader is None:
                    raise ValueError(f"Module spec not found for path {path}")
                module = module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                return getattr(module, class_name)

            recursive_eval = load_module(
                "recursive_eval", repo_path + "/mmdet3d/utils/config.py"
            )

            config_path = (
                repo_path
                + "/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml"
            )

            configs.load(str(config_path), recursive=True)
            cfg = Config(recursive_eval(configs), filename=config_path)
            cfg.model.pretrained = None
            cfg.model.train_cfg = None

            def load_state_dict(
                module: BaseModule, state_dict: dict, prefix: str
            ) -> None:
                module.load_state_dict(
                    {
                        k.replace(prefix, ""): v
                        for k, v in state_dict.items()
                        if k.startswith(prefix)
                    }
                )

            checkpoint = load_torch(ckpt)
            state_dict = checkpoint["state_dict"]

            encoder_backbone = MODELS.build(cfg.model.encoders.camera.backbone)
            load_state_dict(encoder_backbone, state_dict, "encoders.camera.backbone.")

            GeneralizedLSSFPN = load_module(
                "GeneralizedLSSFPN",
                repo_path + "/mmdet3d/models/necks/generalized_lss.py",
            )
            neck_cfg = cfg.model.encoders.camera.neck
            neck_cfg.pop("type")
            encoder_neck = GeneralizedLSSFPN(**neck_cfg)
            load_state_dict(encoder_neck, state_dict, "encoders.camera.neck.")

            vtransform_cfg = cfg.model.encoders.camera.vtransform
            vtransform_cfg.pop("type")
            LSSTransform = load_module(
                "LSSTransform", repo_path + "/mmdet3d/models/vtransforms/lss.py"
            )
            BaseTransform = load_module(
                "BaseTransform", repo_path + "/mmdet3d/models/vtransforms/base.py"
            )

            PatchedLSSTransform = type(
                "PatchedLSSTransform", (LSSTransform, BaseTransform), {}
            )

            LSSTransform.forward = patched_lss_forward
            LSSTransform.get_cam_feats = patched_get_cam_feats
            vtransform = PatchedLSSTransform(**vtransform_cfg)

            load_state_dict(vtransform, state_dict, "encoders.camera.vtransform.")

            GeneralizedResNet = load_module(
                "GeneralizedResNet", repo_path + "/mmdet3d/models/backbones/resnet.py"
            )
            decoder_bb_cfg = cfg.model.decoder.backbone
            decoder_bb_cfg.pop("type")
            decoder_backbone = GeneralizedResNet(**decoder_bb_cfg)
            load_state_dict(decoder_backbone, state_dict, "decoder.backbone.")

            LSSFPN = load_module("LSSFPN", repo_path + "/mmdet3d/models/necks/lss.py")
            decoder_neck_cfg = cfg.model.decoder.neck
            decoder_neck_cfg.pop("type")
            decoder_neck = LSSFPN(**decoder_neck_cfg)
            load_state_dict(decoder_neck, state_dict, "decoder.neck.")

            separate_head_class = load_module(
                "SeparateHead", repo_path + "/mmdet3d/models/heads/bbox/centerpoint.py"
            )
            bbox_coder = load_module(
                "CenterPointBBoxCoder",
                repo_path + "/mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py",
            )
            CenterHead = load_module(
                "CenterHead", repo_path + "/mmdet3d/models/heads/bbox/centerpoint.py"
            )

            bbox_coder_cfg = cfg.model.heads["object"]["bbox_coder"]
            bbox_coder_cfg.pop("type")
            bbox_coder = bbox_coder(**bbox_coder_cfg)
            bbox_coder.onnx_atan2 = onnx_atan2

            CenterHead._topk = patched_topk
            CenterHead.get_task_detections = patched_centerhead_get_task_detections
            CenterHead.circle_nms = staticmethod(patched_circle_nms)

            cfg.model.heads["object"].pop("type", None)
            cfg.model.heads["object"].pop("bbox_coder", None)

            centerhead_cfg = cfg.model.heads["object"]
            centerhead_cfg["separate_head"] = dict(centerhead_cfg["separate_head"])

            centerhead = CenterHead(
                **centerhead_cfg,
                separate_head_cls=separate_head_class,
                bbox_coder=bbox_coder,
            )

            load_state_dict(centerhead, state_dict, "heads.object.")

            models = (
                BEVFusionEncoder1(encoder_backbone, encoder_neck),
                BEVFusionEncoder2(vtransform),
                BEVFusionEncoder3(),
                BEVFusionEncoder4(vtransform),
                BEVFusionDecoder(decoder_backbone, decoder_neck, centerhead),
            )
            for model in models[:-1]:
                model.eval()

            return cls(*models)
