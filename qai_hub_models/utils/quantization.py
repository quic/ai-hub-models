# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader, TensorDataset

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
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
    model: BaseModel,
    input_spec: InputSpec | None = None,
    num_samples: int | None = None,
    dataset_options: dict | None = None,
) -> DatasetEntries:
    """
    Produces a numpy dataset to be used for calibration data of a quantize job.

    If the model has a calibration dataset name set, it will use that dataset.
    Otherwise, it returns the model's sample inputs.

    Parameters:
        model: The model for which to get calibration data.
        input_spec: The input spec of the model. Used to ensure the returned dataset's names
            match the input names of the model.
        num_samples: Number of data samples to use. If not specified, uses
            default specified on dataset.

    Returns:
        Dataset compatible with the format expected by AI Hub.
    """
    calibration_dataset_name = model.calibration_dataset_name()
    if calibration_dataset_name is None:
        assert (
            num_samples is None
        ), "Cannot set num_samples if model doesn't have calibration dataset."
        print(
            "WARNING: Model will be quantized using only a single sample for calibration. "
            + "The quantized model should be only used for performance evaluation, and is unlikely to "
            + "produce reasonable accuracy without additional calibration data."
        )
        return model.sample_inputs(input_spec, use_channel_last_format=False)
    input_spec = input_spec or model.get_input_spec()
    batch_size = get_batch_size(input_spec) or 1
    dataset_options = dataset_options or {}
    dataset = get_dataset_from_name(
        calibration_dataset_name,
        split=DatasetSplit.TRAIN,
        input_spec=input_spec,
        **dataset_options,
    )
    num_samples = num_samples or dataset.default_num_calibration_samples()

    # Round num samples to largest multiple of batch_size less than number requested
    num_samples = (num_samples // batch_size) * batch_size
    print(f"Loading {num_samples} calibration samples.")
    torch_dataset = sample_dataset(dataset, num_samples)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size)
    inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(len(input_spec))]
    for (sample_input, _) in dataloader:
        if isinstance(sample_input, (tuple, list)):
            for i, tensor in enumerate(sample_input):
                inputs[i].append(tensor)
        else:
            inputs[0].append(sample_input)
    return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))


def quantized_folder_deprecation_warning(
    deprecated_package: str, replacement_package: str, precision: Precision
):
    warnings.warn(
        f"""

!!! WARNING !!!
Quantized model package {deprecated_package} is deprecated. Use the equivalent unquantized model package ({replacement_package}) instead.
You can use qai_hub_models.models.{replacement_package}.export and qai_hub_models.models.{replacement_package}.evaluate with the `--precision {str(precision)}` flag to replicate previous behavior of those scripts.

"""
    )
