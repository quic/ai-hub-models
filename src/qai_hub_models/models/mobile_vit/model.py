# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from typing_extensions import Self

from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "apple/mobilevit-small"
MODEL_ASSET_VERSION = 1
TEST_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)


MOBILEVIT_DIM = 256
MOBILEVIT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(MOBILEVIT_DIM),
        transforms.CenterCrop(MOBILEVIT_DIM),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(0)),
    ]
)


class ImagenetMobileVITDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=MOBILEVIT_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_256"

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000


class OpenImagesMobileVITDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=MOBILEVIT_TRANSFORM)


class MobileVIT(ImagenetClassifier):
    """Exportable MobileVIT model, end-to-end."""

    def __init__(
        self,
        net: MobileViTForImageClassification,
        feature_extractor: MobileViTFeatureExtractor,
    ) -> None:
        super().__init__(net, transform_input=False, normalize_input=False)
        self.net = net
        self.feature_extractor = feature_extractor

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        feature_extractor = MobileViTFeatureExtractor.from_pretrained(ckpt_name)
        assert isinstance(feature_extractor, MobileViTFeatureExtractor)
        feature_extractor.size = {"height": 256, "width": 256}
        net = MobileViTForImageClassification.from_pretrained(ckpt_name)
        assert isinstance(net, MobileViTForImageClassification)
        return cls(net, feature_extractor)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(image_tensor, return_dict=False)[0]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, 256, 256),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGE)
        tensor = self.feature_extractor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return dict(image_tensor=[tensor.numpy()])

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetMobileVITDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesMobileVITDataset
