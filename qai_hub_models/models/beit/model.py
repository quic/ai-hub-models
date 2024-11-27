# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import BeitForImageClassification

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "microsoft/beit-base-patch16-224"
MODEL_ASSET_VERSION = 1


class Beit(ImagenetClassifier):

    """Exportable Beit model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):

        return cls(BeitForImageClassification.from_pretrained(ckpt_name))

    def forward(self, image_tensor):

        return self.net(image_tensor, return_dict=False)[0]
