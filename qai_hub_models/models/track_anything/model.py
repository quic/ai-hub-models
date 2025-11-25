# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_numpy,
    load_yaml,
)
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

FIRST_FRAME_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/first_frame_image.npy"
)
FIRST_FRAME_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/first_frame_mask.npy"
)
FIRST_FRAME_F16 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/first_frame_f16.npy"
)
FIRST_FRAME_HIDDEN_STATE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/first_frame_hidden_state.npy"
)
NEXT_FRAME_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_image.npy"
)
NEXT_FRAME_F16 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_f16.npy"
)
NEXT_FRAME_F8 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_f8.npy"
)
NEXT_FRAME_F4 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_f4.npy"
)
NEXT_FRAME_MEMORY_READOUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_memory_readout.npy"
)
NEXT_FRAME_HIDDEN_STATE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs/next_frame_hidden_state.npy"
)

TRACKANYTHING_SOURCE_REPOSITORY = "https://github.com/gaomingqi/Track-Anything.git"
TRACKANYTHING_SOURCE_REPO_COMMIT = "e6e159273790974e04eeea6673f1f93c035005fc"
"""
Remove torch.prod not supported on QNN
Interpolate with area mode is not exportable, changed mode to bilinear
Change 5D to 4D for sigmoid and tanh op to run on NPU
"""
TRACKANYTHING_SOURCE_PATCHES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "patches", "patches.diff"))
]

XMEM_MODEL = CachedWebModelAsset(
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "XMem-s012.pth",
)

with SourceAsRoot(
    TRACKANYTHING_SOURCE_REPOSITORY,
    TRACKANYTHING_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
    source_repo_patches=TRACKANYTHING_SOURCE_PATCHES,
) as repo_path:
    sys.path.append(repo_path + "/tracker")
    from tracker.inference.memory_manager import MemoryManager  # noqa: F401
    from tracker.model.network import XMem


class TrackAnything(BaseModel):
    def __init__(self, model: XMem):
        super().__init__(model)
        self.model: XMem

    @classmethod
    def from_pretrained(cls) -> TrackAnything:
        config = load_yaml(repo_path + "/tracker/config/config.yaml")
        model = XMem(config, XMEM_MODEL.fetch()).eval()
        return cls(model)


class TrackAnythingEncodeKeyWithShrinkage(TrackAnything):
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything EncodeKey model and return key, shrinkage, selection, f16 for given image.

        Parameters
        ----------
            image: torch.Tensor of shape [1, 3, height, width]
                image with value range of [0, 1], RGB channel layout.

        Returns
        -------
            key: torch.Tensor of shape [1, 64, height//16, width//16]
                encoded key
            shrinkage: torch.Tensor of shape [1, 1, height//16, width//16]
                shrinkage key
            selection: torch.Tensor of shape [1, 64, height//16, width//16]
                selection mask
            f16: torch.Tensor of shape [1, 1024, height//16, width//16]
                image features

        """
        image = normalize_image_torchvision(image)

        key, shrinkage, selection, f16, _f8, _f4 = self.model.encode_key(
            image,
            need_ek=True,  # encode_key
            need_sk=True,  # shrinkage_key
        )

        return key, shrinkage, selection, f16

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> InputSpec:
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["key", "shrinkage", "selection", "f16"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_numpy(FIRST_FRAME_IMAGE)
        return {"image": [image]}


class TrackAnythingEncodeValue(TrackAnything):
    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        f16: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything Encode_value model and return generated mask for given points

        Parameters
        ----------
            image: torch.Tensor of shape [1, 3, height, width]
                image with value range of [0, 1], RGB channel layout.
            masks: torch.Tensor of shape [1, height, width]
                mask for first frame
            f16: torch.Tensor of shape [1, 1024, height//16, width//16]
                image feature
            hidden_state: torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        Returns
        -------
            prob: torch.Tensor of shape [2, height, width]
                predicted probabilities
            value: torch.Tensor of shape [1, num_label, 512, height//16, width//16]
                encoded value
            hidden: torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        """
        image = normalize_image_torchvision(image)

        new_mask = torch.cat([1 - mask, mask], dim=0).clamp(1e-7, 1 - 1e-7)
        logits = torch.log(new_mask / (1 - new_mask))
        pred_prob_with_bg = torch.nn.functional.softmax(logits, dim=0)

        value, hidden = self.model.encode_value(
            image,
            f16,
            hidden_state,
            pred_prob_with_bg[1:].unsqueeze(0),
            is_deep_update=True,
        )
        return pred_prob_with_bg, value, hidden

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> InputSpec:
        return {
            "image": ((batch_size, 3, height, width), "float32"),
            "mask": ((batch_size, height, width), "float32"),
            "f16": ((batch_size, 1024, height // 16, width // 16), "float32"),
            "hidden_state": ((batch_size, 1, 64, height // 16, width // 16), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["masks", "value", "hidden"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_numpy(FIRST_FRAME_IMAGE)
        mask = load_numpy(FIRST_FRAME_MASK)
        f16 = load_numpy(FIRST_FRAME_F16)
        hidden_state = load_numpy(FIRST_FRAME_HIDDEN_STATE)
        return {
            "image": [image],
            "mask": [mask],
            "f16": [f16],
            "hidden_state": [hidden_state],
        }


class TrackAnythingEncodeKeyWithoutShrinkage(TrackAnything):
    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything Encode_Key model and return key, selection, f16, f8, f4 for given image.

        Parameters
        ----------
            image: torch.Tensor of shape [1, 3, height, width]
                image with value range of [0, 1], RGB channel layout.

        Returns
        -------
            key: torch.Tensor of shape [1, 64, height//16, width//16]
                encoded key
            selection: torch.Tensor of shape [1, 64, height//16, width//16]
                selection mask
            f16: torch.Tensor of shape [1, 1024, height//16, width//16]
                image features
            f8: torch.Tensor of shape [1, 512, height//8, width//8]
                image features
            f4: torch.Tensor of shape [1, 256, height//4, width//4]
                image features
        """
        image = normalize_image_torchvision(image)

        key, _, selection, f16, f8, f4 = self.model.encode_key(
            image,
            need_ek=True,
            need_sk=False,
        )
        return key, selection, f16, f8, f4

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> InputSpec:
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["key", "selection", "f16", "f8", "f4"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_numpy(NEXT_FRAME_IMAGE)
        return {"image": [image]}


class TrackAnythingSegment(TrackAnything):
    def forward(
        self,
        f16: torch.Tensor,
        f8: torch.Tensor,
        f4: torch.Tensor,
        memory_readout: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run TrackAnything model and return generated mask for given points

        Parameters
        ----------
            f16: torch.Tensor of shape [1, 1024, height//16, width//16]
                image features
            f8: torch.Tensor of shape [1, 512, height//8, width//8]
                image features
            f4: torch.Tensor of shape [1, 256, height//4, width//4]
                image features
            memory_readout: torch.Tensor of shape [1, num_label, 512, height//16, width//16]
                memory matched with current key and selection
            hidden_state: torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        Returns
        -------
            prob: torch.Tensor of shape [2, height, width]
                predicted probabilities
            hidden: torch.Tensor of shape [1, num_label, 64, height//16, width//16]

        """
        multi_scale_features = (f16, f8, f4)

        # segment the current frame
        hidden, _pred_logits_with_bg, pred_prob_with_bg = self.model.segment(
            multi_scale_features,
            memory_readout,
            hidden_state,
            h_out=True,
            strip_bg=False,
        )
        # remove batch dim
        pred_prob_with_bg = pred_prob_with_bg[0]

        return pred_prob_with_bg, hidden

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 576,
    ) -> InputSpec:
        return {
            "f16": ((batch_size, 1024, height // 16, width // 16), "float32"),
            "f8": ((batch_size, 512, height // 8, width // 8), "float32"),
            "f4": ((batch_size, 256, height // 4, width // 4), "float32"),
            "memory_readout": (
                (batch_size, 1, 512, height // 16, width // 16),
                "float32",
            ),
            "hidden_state": ((batch_size, 1, 64, height // 16, width // 16), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["masks", "hidden"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        f16 = load_numpy(NEXT_FRAME_F16)
        f8 = load_numpy(NEXT_FRAME_F8)
        f4 = load_numpy(NEXT_FRAME_F4)
        memory_readout = load_numpy(NEXT_FRAME_MEMORY_READOUT)
        hidden_state = load_numpy(NEXT_FRAME_HIDDEN_STATE)
        return {
            "f16": [f16],
            "f8": [f8],
            "f4": [f4],
            "memory_readout": [memory_readout],
            "hidden_state": [hidden_state],
        }


@CollectionModel.add_component(TrackAnythingEncodeKeyWithShrinkage)
@CollectionModel.add_component(TrackAnythingEncodeValue)
@CollectionModel.add_component(TrackAnythingEncodeKeyWithoutShrinkage)
@CollectionModel.add_component(TrackAnythingSegment)
class TrackAnythingWrapper(CollectionModel):
    def __init__(
        self,
        EncodeKeyWithShrinkage: TrackAnythingEncodeKeyWithShrinkage,
        EncodeValue: TrackAnythingEncodeValue,
        EncodeKeyWithoutShrinkage: TrackAnythingEncodeKeyWithoutShrinkage,
        Segment: TrackAnythingSegment,
        config: dict,
    ) -> None:
        super().__init__(
            EncodeKeyWithShrinkage, EncodeValue, EncodeKeyWithoutShrinkage, Segment
        )
        self.EncodeKeyWithShrinkage = EncodeKeyWithShrinkage
        self.EncodeValue = EncodeValue
        self.EncodeKeyWithoutShrinkage = EncodeKeyWithoutShrinkage
        self.Segment = Segment
        self.config = config

    @classmethod
    def from_pretrained(cls) -> TrackAnythingWrapper:
        config = load_yaml(repo_path + "/tracker/config/config.yaml")
        model = XMem(config, XMEM_MODEL.fetch()).eval()
        EncodeKeyWithShrinkage = TrackAnythingEncodeKeyWithShrinkage(model)
        EncodeValue = TrackAnythingEncodeValue(model)
        EncodeKeyWithoutShrinkage = TrackAnythingEncodeKeyWithoutShrinkage(model)
        Segment = TrackAnythingSegment(model)
        return cls(
            EncodeKeyWithShrinkage,
            EncodeValue,
            EncodeKeyWithoutShrinkage,
            Segment,
            config,
        )
