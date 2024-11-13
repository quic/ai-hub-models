# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import torch
from qai_hub.client import DatasetEntries, Device, QuantizeDtype
from torch.utils.data import DataLoader, TensorDataset

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.protocols import HubModelProtocol
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_torch
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.inference import make_hub_dataset_entries
from qai_hub_models.utils.input_spec import InputSpec

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
    torch_dataset = sample_dataset(
        get_dataset_from_name(dataset_name, split=DatasetSplit.TRAIN), num_samples
    )
    torch_samples = tuple(
        [torch_dataset[i][j].unsqueeze(0).numpy() for i in range(len(torch_dataset))]
        for j in range(len(input_spec))
    )
    return make_hub_dataset_entries(torch_samples, list(input_spec.keys()))


class HubQuantizableMixin(HubModelProtocol):
    """
    Mixin to attach to model classes that will be quantized using AI Hub quantize job.
    """

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_io"
        if target_runtime == TargetRuntime.TFLITE:
            # uint8 is the easiest I/O type for integration purposes,
            # especially for image applications. Images are always
            # uint8 RGB when coming from disk or a camera.
            #
            # Uint8 has not been thoroughly tested with other paths,
            # so it is enabled only for TF Lite today.
            quantization_flags += " --quantize_io_type uint8"
        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, other_compile_options, device
            )
            + quantization_flags
        )

    def get_quantize_options(self) -> str:
        return ""

    @staticmethod
    def get_weights_dtype() -> QuantizeDtype:
        return QuantizeDtype.INT8

    @staticmethod
    def get_activations_dtype() -> QuantizeDtype:
        return QuantizeDtype.INT8
