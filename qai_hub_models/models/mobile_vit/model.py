# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "apple/mobilevit-small"
MODEL_ASSET_VERSION = 1
TEST_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)


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
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        feature_extractor = MobileViTFeatureExtractor.from_pretrained(ckpt_name)
        assert isinstance(feature_extractor, MobileViTFeatureExtractor)
        feature_extractor.size = {"height": 224, "width": 224}
        net = MobileViTForImageClassification.from_pretrained(ckpt_name)
        assert isinstance(net, MobileViTForImageClassification)
        return cls(net, feature_extractor)

    def forward(self, image_tensor):
        return self.net(image_tensor, return_dict=False)[0]

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        return {"image_tensor": ((batch_size, 3, 256, 256), "float32")}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGE)
        tensor = self.feature_extractor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return dict(image_tensor=[tensor.numpy()])
