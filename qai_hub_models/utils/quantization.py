# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader, TensorDataset

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_torch
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

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
    tensor_dataset = TensorDataset(img_tensor)
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


def get_calibration_data(
    input_spec: InputSpec, dataset_name: str, num_samples: int
) -> DatasetEntries:
    """
    Produces a numpy dataset to be used for calibration data of a quantize job.

    Parameters:
        input_spec: The input spec of the model. Used to ensure the returned dataset's names
            match the input names of the model.
        dataset_name: Name of the dataset to sample from.
        num_samples: Number of data samples to use.

    Returns:
        Dataset compatible with the format expected by AI Hub.
    """
    batch_size = get_batch_size(input_spec)

    # Round num samples to largest multiple of batch_size less than number requested
    num_samples = (num_samples // batch_size) * batch_size
    torch_dataset = sample_dataset(
        get_dataset_from_name(dataset_name, split=DatasetSplit.TRAIN), num_samples
    )
    dataloader = DataLoader(torch_dataset, batch_size=batch_size)
    inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(len(input_spec))]
    for (sample_input, _) in dataloader:
        if isinstance(sample_input, tuple):
            for i, tensor in enumerate(sample_input):
                inputs[i].append(tensor)
        else:
            inputs[0].append(sample_input)
    return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
