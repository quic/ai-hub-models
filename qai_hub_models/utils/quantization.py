# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_torch

DATA_ID = "image_quantziation_samples"
DATA_VERSION = 1

IMAGE_QUANTIZATION_SAMPLES = CachedWebDatasetAsset(
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/datasets/image_quantization_samples.pt",
    DATA_ID,
    DATA_VERSION,
    "image_quantization_samples.pt",
)


def make_image_sample_data_loader() -> DataLoader:
    img_tensor = get_image_quantization_samples()
    tensor_dataset = torch.utils.data.TensorDataset(img_tensor)
    return DataLoader(tensor_dataset, batch_size=32)


def get_image_quantization_samples(
    quantization_samples_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Loads a tensor of sample input image data from the specified path.
    This data is intended to be used for post-training quantization.

    If no path is provided, the method returns a default tensor containing
    data from images fetched from the Google OpenImages dataset.

    The default tensor has shape (50, 3, 224, 224). Here is the code to produce
    the default tensor:

    ```
    import fiftyone.zoo as foz
    from PIL import Image
    import torch
    from qai_hub_models.models._shared.imagenet_classifier.app import preprocess_image

    image_dataset = foz.load_models_dataset(
        "open-images-v6",
        split="validation",
        max_samples=50,
        shuffle=True,
    )

    tensors = []
    for sample in image_dataset:
        img = Image.open(sample.filepath)
        tensors.append(preprocess_image(img))

    final_tensor = torch.cat(tensors, dim=0)

    torch.save(final_tensor, "imagenet_quantization_samples.pt")
    ```
    """
    path = IMAGE_QUANTIZATION_SAMPLES.fetch(extract=False)
    return load_torch(quantization_samples_path or path)
