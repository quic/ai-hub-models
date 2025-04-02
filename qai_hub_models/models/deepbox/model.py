# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.yolov3.model import YoloV3
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_torch,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
DEEPBOX_SOURCE_REPOSITORY = "https://github.com/skhadem/3D-BoundingBox.git"
DEEPBOX_SOURCE_REPO_COMMIT = "40ba3b60fd550ff33d5c1b307212dad80149e525"

VGG_WEIGHTS_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "epoch_10.pkl"
)
DEFAULT_YOLO_WEIGHTS = "yolov3-tiny.pt"


class Yolo2DDetection(YoloV3):
    """
    Exportable YoloV3 bounding box detector, end-to-end.

    Hand detection model. Input is an image, output is
    [bounding boxes & keypoints, box & keypoint scores]
    """

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 640,
    ):
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return super()._sample_inputs_impl(input_spec or self.get_input_spec())

    @staticmethod
    def calibration_dataset_name() -> str | None:
        return None


class VGG3DDetection(BaseModel):
    """
    Hand landmark detector model. Input is an image cropped to the hand. The hand must be upright
    and un-tilted in the frame. Returns [landmark_scores, prob_is_right_hand, landmarks]
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "DEFAULT") -> VGG3DDetection:  # type: ignore[override]
        with SourceAsRoot(
            DEEPBOX_SOURCE_REPOSITORY,
            DEEPBOX_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            if ckpt_path == "DEFAULT":
                ckpt_path = str(VGG_WEIGHTS_ASSET.fetch())

            find_replace_in_repo(
                repo_path,
                "library/Math.py",
                "A = np.zeros([4,3], dtype=np.float)",
                "A = np.zeros([4,3], dtype=float)",
            )

            from torch_lib import Model
            from torchvision.models import vgg

            my_vgg = vgg.vgg19_bn(pretrained=True)
            vgg_model = Model.Model(features=my_vgg.features, bins=2)
            checkpoint = load_torch(ckpt_path)
            vgg_model.load_state_dict(checkpoint["model_state_dict"])
        return cls(vgg_model)

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        return out[0], out[1], out[2]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
    ):
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["orient", "conf", "dim"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


@CollectionModel.add_component(Yolo2DDetection)
@CollectionModel.add_component(VGG3DDetection)
class DeepBox(PretrainedCollectionModel):
    def __init__(self, yolo_2d_det: Yolo2DDetection, vgg_3d_det: VGG3DDetection):
        super().__init__(yolo_2d_det, vgg_3d_det)
        self.yolo_2d_det = yolo_2d_det
        self.vgg_3d_det = vgg_3d_det

    @classmethod
    def from_pretrained(
        cls,
        yolo_ckpt: str = DEFAULT_YOLO_WEIGHTS,
        vgg_ckpt_path: str = "DEFAULT",
    ) -> DeepBox:
        yolo = Yolo2DDetection.from_pretrained(yolo_ckpt)
        vgg_net = VGG3DDetection.from_pretrained(vgg_ckpt_path)
        return cls(yolo, vgg_net)
