# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qai_hub_models.models.common import Precision, SampleInputsType
from qai_hub_models.models.mediapipe_hand.model import HandDetector
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import InputSpec

# ------------------------------------------------------------------------------------
# Constants & simple assets
# ------------------------------------------------------------------------------------
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
SAMPLE_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "thumbs_up.jpg"
)

GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

BATCH_SIZE = 1


class PalmDetector(HandDetector):
    """
    Hand detection model. Input is an image, output is
    [bounding boxes & keypoints, box & keypoint scores]
    """

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    @staticmethod
    def calibration_dataset_name() -> str:
        return "hagrid_palmdetector"


class HandLandmarkDetector(BaseModel):
    """Hand landmark regression network.

    Returns :
      - landmarks (63-d: 21 * (x,y,z) local coords)
      - scores (confidence)
      - lr (left/right handedness score)
      - world_landmarks (63-d world coords)
    """

    def __init__(self):
        super().__init__()
        # ---- Layers ----
        self.conv_0 = nn.Conv2d(
            3,
            24,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_1 = nn.Conv2d(
            24,
            24,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=24,
            bias=True,
        )
        self.conv_2 = nn.Conv2d(
            24,
            16,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_3 = nn.Conv2d(
            16,
            64,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_4 = nn.Conv2d(
            64,
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            dilation=(1, 1),
            groups=64,
            bias=True,
        )
        self.conv_5 = nn.Conv2d(
            64,
            24,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_6 = nn.Conv2d(
            24,
            144,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_7 = nn.Conv2d(
            144,
            144,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=144,
            bias=True,
        )
        self.conv_8 = nn.Conv2d(
            144,
            24,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_9 = nn.Conv2d(
            24,
            144,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_10 = nn.Conv2d(
            144,
            144,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=0,
            dilation=(1, 1),
            groups=144,
            bias=True,
        )
        self.conv_11 = nn.Conv2d(
            144,
            40,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_12 = nn.Conv2d(
            40,
            240,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_13 = nn.Conv2d(
            240,
            240,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=240,
            bias=True,
        )
        self.conv_14 = nn.Conv2d(
            240,
            40,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_15 = nn.Conv2d(
            40,
            240,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_16 = nn.Conv2d(
            240,
            240,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            dilation=(1, 1),
            groups=240,
            bias=True,
        )
        self.conv_17 = nn.Conv2d(
            240,
            80,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_18 = nn.Conv2d(
            80,
            480,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_19 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=480,
            bias=True,
        )
        self.conv_20 = nn.Conv2d(
            480,
            80,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_21 = nn.Conv2d(
            80,
            480,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_22 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=480,
            bias=True,
        )
        self.conv_23 = nn.Conv2d(
            480,
            80,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_24 = nn.Conv2d(
            80,
            480,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_25 = nn.Conv2d(
            480,
            480,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=480,
            bias=True,
        )
        self.conv_26 = nn.Conv2d(
            480,
            112,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_27 = nn.Conv2d(
            112,
            672,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_28 = nn.Conv2d(
            672,
            672,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=672,
            bias=True,
        )
        self.conv_29 = nn.Conv2d(
            672,
            112,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_30 = nn.Conv2d(
            112,
            672,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_31 = nn.Conv2d(
            672,
            672,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=672,
            bias=True,
        )
        self.conv_32 = nn.Conv2d(
            672,
            112,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_33 = nn.Conv2d(
            112,
            672,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_34 = nn.Conv2d(
            672,
            672,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=0,
            dilation=(1, 1),
            groups=672,
            bias=True,
        )
        self.conv_35 = nn.Conv2d(
            672,
            192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_36 = nn.Conv2d(
            192,
            1152,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_37 = nn.Conv2d(
            1152,
            1152,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1152,
            bias=True,
        )
        self.conv_38 = nn.Conv2d(
            1152,
            192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_39 = nn.Conv2d(
            192,
            1152,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_40 = nn.Conv2d(
            1152,
            1152,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1152,
            bias=True,
        )
        self.conv_41 = nn.Conv2d(
            1152,
            192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_42 = nn.Conv2d(
            192,
            1152,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_43 = nn.Conv2d(
            1152,
            1152,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1152,
            bias=True,
        )
        self.conv_44 = nn.Conv2d(
            1152,
            192,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_45 = nn.Conv2d(
            192,
            1152,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )
        self.conv_46 = nn.Conv2d(
            1152,
            1152,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1152,
            bias=True,
        )
        # Heads
        self.linear_0 = nn.Linear(in_features=1152, out_features=1, bias=True)
        self.linear_1 = nn.Linear(in_features=1152, out_features=1, bias=True)
        self.linear_2 = nn.Linear(in_features=1152, out_features=63, bias=True)
        self.linear_3 = nn.Linear(in_features=1152, out_features=63, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the hand landmark regression backbone + heads.

        Parameters
        ----------
        x
          Input image tensor of shape (1, 3, 224, 224).

        Returns
        -------
        landmarks (t_93): (1, 63)       21 * (x,y,z) local coords
        scores    (t_96): (1, 1)        confidence score in [0,1]
        lr        (t_95): (1, 1)        left/right handedness score in [0,1]
        world_landmarks (t_94): (1, 63) 21 * (x,y,z) world coords

        """
        _x = F.pad(x, (0, 1, 0, 1))
        t_1 = self.conv_0(_x)
        t_2 = torch.clamp(t_1, min=0.0, max=6.0)
        _x = F.pad(t_2, (1, 1, 1, 1))
        t_3 = self.conv_1(_x)
        t_4 = torch.clamp(t_3, min=0.0, max=6.0)
        t_5 = self.conv_2(t_4)
        t_6 = self.conv_3(t_5)
        t_7 = torch.clamp(t_6, min=0.0, max=6.0)
        _x = F.pad(t_7, (0, 1, 0, 1))
        t_8 = self.conv_4(_x)
        t_9 = torch.clamp(t_8, min=0.0, max=6.0)
        t_10 = self.conv_5(t_9)
        t_11 = self.conv_6(t_10)
        t_12 = torch.clamp(t_11, min=0.0, max=6.0)
        _x = F.pad(t_12, (1, 1, 1, 1))
        t_13 = self.conv_7(_x)
        t_14 = torch.clamp(t_13, min=0.0, max=6.0)
        t_15 = self.conv_8(t_14)
        t_16 = t_15 + t_10
        t_17 = self.conv_9(t_16)
        t_18 = torch.clamp(t_17, min=0.0, max=6.0)
        _x = F.pad(t_18, (1, 2, 1, 2))
        t_19 = self.conv_10(_x)
        t_20 = torch.clamp(t_19, min=0.0, max=6.0)
        t_21 = self.conv_11(t_20)
        t_22 = self.conv_12(t_21)
        t_23 = torch.clamp(t_22, min=0.0, max=6.0)
        _x = F.pad(t_23, (2, 2, 2, 2))
        t_24 = self.conv_13(_x)
        t_25 = torch.clamp(t_24, min=0.0, max=6.0)
        t_26 = self.conv_14(t_25)
        t_27 = t_26 + t_21
        t_28 = self.conv_15(t_27)
        t_29 = torch.clamp(t_28, min=0.0, max=6.0)
        _x = F.pad(t_29, (0, 1, 0, 1))
        t_30 = self.conv_16(_x)
        t_31 = torch.clamp(t_30, min=0.0, max=6.0)
        t_32 = self.conv_17(t_31)
        t_33 = self.conv_18(t_32)
        t_34 = torch.clamp(t_33, min=0.0, max=6.0)
        _x = F.pad(t_34, (1, 1, 1, 1))
        t_35 = self.conv_19(_x)
        t_36 = torch.clamp(t_35, min=0.0, max=6.0)
        t_37 = self.conv_20(t_36)
        t_38 = t_37 + t_32
        t_39 = self.conv_21(t_38)
        t_40 = torch.clamp(t_39, min=0.0, max=6.0)
        _x = F.pad(t_40, (1, 1, 1, 1))
        t_41 = self.conv_22(_x)
        t_42 = torch.clamp(t_41, min=0.0, max=6.0)
        t_43 = self.conv_23(t_42)
        t_44 = t_43 + t_38
        t_45 = self.conv_24(t_44)
        t_46 = torch.clamp(t_45, min=0.0, max=6.0)
        _x = F.pad(t_46, (2, 2, 2, 2))
        t_47 = self.conv_25(_x)
        t_48 = torch.clamp(t_47, min=0.0, max=6.0)
        t_49 = self.conv_26(t_48)
        t_50 = self.conv_27(t_49)
        t_51 = torch.clamp(t_50, min=0.0, max=6.0)
        _x = F.pad(t_51, (2, 2, 2, 2))
        t_52 = self.conv_28(_x)
        t_53 = torch.clamp(t_52, min=0.0, max=6.0)
        t_54 = self.conv_29(t_53)
        t_55 = t_54 + t_49
        t_56 = self.conv_30(t_55)
        t_57 = torch.clamp(t_56, min=0.0, max=6.0)
        _x = F.pad(t_57, (2, 2, 2, 2))
        t_58 = self.conv_31(_x)
        t_59 = torch.clamp(t_58, min=0.0, max=6.0)
        t_60 = self.conv_32(t_59)
        t_61 = t_60 + t_55
        t_62 = self.conv_33(t_61)
        t_63 = torch.clamp(t_62, min=0.0, max=6.0)
        _x = F.pad(t_63, (1, 2, 1, 2))
        t_64 = self.conv_34(_x)
        t_65 = torch.clamp(t_64, min=0.0, max=6.0)
        t_66 = self.conv_35(t_65)
        t_67 = self.conv_36(t_66)
        t_68 = torch.clamp(t_67, min=0.0, max=6.0)
        _x = F.pad(t_68, (2, 2, 2, 2))
        t_69 = self.conv_37(_x)
        t_70 = torch.clamp(t_69, min=0.0, max=6.0)
        t_71 = self.conv_38(t_70)
        t_72 = t_71 + t_66
        t_73 = self.conv_39(t_72)
        t_74 = torch.clamp(t_73, min=0.0, max=6.0)
        _x = F.pad(t_74, (2, 2, 2, 2))
        t_75 = self.conv_40(_x)
        t_76 = torch.clamp(t_75, min=0.0, max=6.0)
        t_77 = self.conv_41(t_76)
        t_78 = t_77 + t_72
        t_79 = self.conv_42(t_78)
        t_80 = torch.clamp(t_79, min=0.0, max=6.0)
        _x = F.pad(t_80, (2, 2, 2, 2))
        t_81 = self.conv_43(_x)
        t_82 = torch.clamp(t_81, min=0.0, max=6.0)
        t_83 = self.conv_44(t_82)
        t_84 = t_83 + t_78
        t_85 = self.conv_45(t_84)
        t_86 = torch.clamp(t_85, min=0.0, max=6.0)
        _x = F.pad(t_86, (1, 1, 1, 1))
        t_87 = self.conv_46(_x)
        t_88 = torch.clamp(t_87, min=0.0, max=6.0)
        t_89 = t_88.mean(dim=(2, 3))
        t_90 = t_89
        t_91 = self.linear_0(t_90)
        t_92 = self.linear_1(t_90)
        t_93 = self.linear_2(t_90)
        t_94 = self.linear_3(t_90)
        t_95 = torch.sigmoid(t_92)
        t_96 = torch.sigmoid(t_91)

        return (t_93, t_96, t_95, t_94)

    @classmethod
    def from_pretrained(
        cls, landmark_detector_weights_path: str | None = None
    ) -> HandLandmarkDetector:
        hand_regressor = HandLandmarkDetector()
        if landmark_detector_weights_path is None:
            landmark_detector_weights_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, "hand_landmark_full_weights.pth"
            ).fetch()
        landmark_detector_weight = torch.load(
            landmark_detector_weights_path, map_location="cpu", weights_only=False
        )
        hand_regressor.load_state_dict(landmark_detector_weight, strict=True)
        hand_regressor.eval()
        return hand_regressor

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, 224, 224), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["landmarks", "scores", "lr", "world_landmarks"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        numpy_inputs = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "sample_landmark_inputs.npy"
        )
        return {"image": [load_numpy(numpy_inputs)]}

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    @staticmethod
    def calibration_dataset_name() -> str:
        return "hagrid_handlandmark"


# ------------------------------------------------------------------------------------
# Inner MLP used by GestureEmbedder
# ------------------------------------------------------------------------------------
class _InnerGestureMLP(nn.Module):
    """
    Inner submodel
    Takes a 64D input (local hand coords + handedness scalar)
    and processes it through the 7-block residual MLP.
    """

    def __init__(self, dropout_p: float = 0.1, bn_eps: float = 1e-3):
        super().__init__()
        self.fc0 = nn.Linear(64, 128, bias=True)
        self.bn0 = nn.BatchNorm1d(128, eps=bn_eps)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout_p)
        self.fcs = nn.ModuleList([nn.Linear(128, 128, bias=True) for _ in range(7)])
        self.bns_after_add = nn.ModuleList(
            [nn.BatchNorm1d(128, eps=bn_eps, momentum=0.01) for _ in range(6)]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x64: torch.Tensor) -> torch.Tensor:
        """Processes the 64D input through the residual stack."""
        z0 = self.fc0(x64)
        # Input to first residual FC is post-BN/Act/Drop of the initial dense layer
        h = self.drop(self.act(self.bn0(z0)))

        # Start of residual path accumulation
        s = z0 + self.fcs[0](h)

        # Loop through add_2 to add_6
        for i in range(1, 6):
            # 'h_i' is the post-BN/Act/Drop of the previous 's' result
            h_i = self.drop(self.act(self.bns_after_add[i - 1](s)))
            s = s + self.fcs[i](h_i)

        # Final block (add_7): post-BN/Act/Drop then final linear add
        h6 = self.drop(self.act(self.bns_after_add[5](s)))
        return s + self.fcs[6](h6)


# ------------------------------------------------------------------------------------
# GestureEmbedder+Classifier
# ------------------------------------------------------------------------------------
class CannedGestureClassifier(BaseModel):
    """
    Top-level model configured specifically with preprocessed hand-only and handedness input.
    Combined mediapipe gesture embedder and classifier
    Input is preprocessed hand landmarks with handedness
    returns an array of scores indicating gesture probabilities.
    """

    def __init__(
        self,
        dropout_p: float = 0.1,
        bn_eps: float = 1e-3,
        num_classes: int = 8,
    ):
        super().__init__()
        self.inner_mlp = _InnerGestureMLP(dropout_p=dropout_p, bn_eps=bn_eps)
        self.bn_out = nn.BatchNorm1d(128, eps=bn_eps, momentum=0.01)
        self.act_out = nn.SiLU()
        self.drop_out = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(128, 128, bias=True)
        self.gamma = nn.Parameter(torch.ones(128))
        self.beta = nn.Parameter(torch.zeros(128))
        self.fc = nn.Linear(128, num_classes, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x64_a: torch.Tensor, x64_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gesture embedder/classifier.

        Parameters
        ----------
        x64_a
            Tensor of shape (1, 64). Preprocessed features for branch A
            corresponding to the original (non-mirrored) hand. Each row contains:
            63 normalized landmark features (flattened from 21 * 3),
            and 1 handedness scalar in {0.0, 1.0}.
        x64_b
            Tensor of shape (1, 64). Preprocessed features for branch B
            corresponding to the mirrored hand (x-flip). Each row contains:
            63 normalized landmark features (flattened from 21 * 3),
            and 1 handedness scalar in {0.0, 1.0}, inverted relative to x64_a.

        Returns
        -------
        logits
            Probabilities (softmax over classes) of shape (1, 8).
        """
        # Branch A (already preprocessed)
        a = self.inner_mlp(x64_a)
        # Branch B (mirrored)
        b = self.inner_mlp(x64_b)
        # Sum branches → BN→SiLU→Dropout→FC to get embedding
        s = a + b
        h = self.drop_out(self.act_out(self.bn_out(s)))
        emb = self.fc_out(h)  # shape: (N, 128)
        # Classifier
        x = emb * self.gamma + self.beta
        x = F.relu(x, inplace=True)
        logits = self.fc(x)
        return F.softmax(logits, dim=-1)

    @classmethod
    def from_pretrained(
        cls,
        embedder_weights_path: str | None = None,
        classifier_weights_path: str | None = None,
        **kwargs,
    ) -> CannedGestureClassifier:
        model = cls(**kwargs)

        if embedder_weights_path is None:
            embedder_weights_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, "gesture_embedder.pth"
            ).fetch()
        embedder_weights = torch.load(embedder_weights_path, map_location="cpu")
        model.load_state_dict(embedder_weights, strict=False)

        if classifier_weights_path is None:
            classifier_weights_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, "gesture_classifier.pth"
            ).fetch()
        classifier_weights = torch.load(classifier_weights_path, map_location="cpu")
        model.load_state_dict(classifier_weights, strict=False)
        model.eval()
        return model

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the gesture classifier.
        This can be used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "hand": ((batch_size, 64), "float32"),
            "mirrored_hand": ((batch_size, 64), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["Identity"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        hand_numpy_inputs = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "preprocessed_hand.npy"
        )
        handedness_numpy_inputs = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "preprocessed_mirrored_hand.npy"
        )
        return {
            "hand": [load_numpy(hand_numpy_inputs)],
            "mirrored_hand": [load_numpy(handedness_numpy_inputs)],
        }

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    @staticmethod
    def calibration_dataset_name() -> str:
        return "hagrid_gesturerecognizer"


# ------------------------------------------------------------------------------------
# Collection wiring
# ------------------------------------------------------------------------------------


@CollectionModel.add_component(PalmDetector)
@CollectionModel.add_component(HandLandmarkDetector)
@CollectionModel.add_component(CannedGestureClassifier)
class MediaPipeHandGesture(PretrainedCollectionModel):
    """Collection model that wires detector, landmark, embedder, classifier."""

    def __init__(
        self,
        palm_detector: PalmDetector,
        hand_landmark_detector: HandLandmarkDetector,
        hand_gesture_classifier: CannedGestureClassifier,
    ) -> None:
        super().__init__(
            palm_detector,
            hand_landmark_detector,
            hand_gesture_classifier,
        )
        self.palm_detector = palm_detector
        self.hand_landmark_detector = hand_landmark_detector
        self.gesture_classifier = hand_gesture_classifier

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepalm.pth",
        detector_anchors: str = "anchors_palm.npy",
    ) -> MediaPipeHandGesture:
        return cls(
            PalmDetector.from_pretrained(detector_weights, detector_anchors),
            HandLandmarkDetector.from_pretrained(),
            CannedGestureClassifier.from_pretrained(),
        )
