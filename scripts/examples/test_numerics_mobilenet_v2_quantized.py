# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run it with pytest --on-device
"""
from typing import Tuple

import numpy as np
import pytest
import qai_hub as hub
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from qai_hub_models.datasets.imagenette import ImagenetteDataset
from qai_hub_models.models.mobilenet_v2_quantized.model import MobileNetV2Quantizable
from qai_hub_models.utils.inference import compile_zoo_model_to_hub
from qai_hub_models.utils.measurement import get_model_size_mb


def on_device(func):
    # Skip tests if '--on-device' is not in the command line arguments
    return pytest.mark.skipif(
        "'--on-device' not in sys.argv", reason="needs --on-device option to run"
    )(func)


@pytest.fixture(scope="module")
def data_loaders():
    dataset = ImagenetteDataset()
    calib_len = int(0.1 * len(dataset))
    test_len = len(dataset) - calib_len
    # Deterministic random split
    calib_dataset, test_dataset = random_split(
        dataset, [calib_len, test_len], generator=torch.Generator().manual_seed(42)
    )
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return calib_loader, test_loader


@pytest.fixture(scope="module")
def test_data(data_loaders) -> Tuple[torch.Tensor, torch.Tensor, hub.Dataset]:
    calib_loader, test_loader = data_loaders
    num_test = 1000

    img_batches, label_batches = [], []
    total_samples = 0
    for images, labels in tqdm(test_loader):
        img_batches.append(images)
        label_batches.append(labels)
        total_samples += images.size(0)
        if total_samples >= 1000:
            break
    img_test = torch.cat(img_batches, dim=0)[:num_test]
    label_test = torch.cat(label_batches, dim=0)[:num_test]
    input_name = list(
        MobileNetV2Quantizable.from_pretrained(aimet_encodings=None)
        .get_input_spec()
        .keys()
    )[0]
    data_entries = {input_name: np.split(img_test.numpy(), img_test.shape[0])}
    hub_ds = hub.upload_dataset(data_entries)
    return img_test, label_test, hub_ds


def test_dataloader_is_deterministic(data_loaders):
    """Test that the calibration-test split and the loading are deterministic"""
    calib_loader, test_loader = data_loaders
    img, labels = next(iter(calib_loader))
    expected_calib_labels = [701, 569, 482, 571, 482]
    assert labels[:5].tolist() == expected_calib_labels

    expected_test_labels = [569, 0, 217, 571, 701]
    img, labels = next(iter(test_loader))
    assert labels[:5].tolist() == expected_test_labels


@pytest.fixture(scope="module")
def quantized_model(data_loaders, test_data):
    """
    Create encoding from calibration data and returned quantized model with
    validated off-target accuracy computed on QuantSim
    """
    img_test, label_test, hub_dataset = test_data
    calib_loader, test_loader = data_loaders
    model = MobileNetV2Quantizable.from_pretrained(aimet_encodings=None)

    # Calibration in quantization
    num_calib_batches = 3
    calib_accuracy = model.quantize(
        calib_loader, num_calib_batches, evaluator=model.get_evaluator()
    )
    np.testing.assert_allclose(0.76, calib_accuracy, atol=0.01)

    # QuantSim evaluation on eval set
    evaluator = model.get_evaluator()

    batch_size = 32
    for i in tqdm(list(range(0, img_test.size(0), batch_size)), desc="QuantSim eval"):
        img_batch = img_test[i : i + batch_size]
        label_batch = label_test[i : i + batch_size]

        sim_out = model(img_batch).detach()
        evaluator.add_batch(sim_out, label_batch)

    sim_acc = evaluator.get_accuracy_score()
    print(f"{sim_acc=}")
    np.testing.assert_allclose(0.78125, sim_acc, atol=0.01)
    return model


@on_device
@pytest.mark.parametrize(
    "target_runtime,hub_needs_calib_data,expected_size_mb,expected_acc",
    [
        ("onnx-tflite", False, 3.806, 0),
        ("torch-tflite", False, 7.0891, 0.719),
        ("onnx-qnn", False, 3.844, 0.76),
        ("torch-qnn", True, 3.82, 0.7618),
    ],
)
def test_make_encoding_w8a8_accuracy(
    quantized_model,
    data_loaders,
    target_runtime,
    hub_needs_calib_data,
    expected_size_mb,
    expected_acc,
    test_data,
):
    """
    1. Export and compile quantized_model on Hub.
    2. Run inference on Hub on test.

    Note: We don't run profile job to get perf here but leave that to the score card.
    """
    model = quantized_model

    img_test, label_test, hub_dataset = test_data
    calib_loader, test_loader = data_loaders

    # calibration data
    calibration_data = None
    if hub_needs_calib_data:
        # AIMET export has missing encoding and needs calibration data
        num_calib_batches = 3
        calib_imgs = []
        for b, (img_calib, labels) in enumerate(iter(calib_loader)):
            if b >= num_calib_batches:
                break
            img_np = img_calib.numpy()
            calib_imgs.extend(np.split(img_np, img_np.shape[0]))
        calibration_data = {list(model.get_input_spec().keys())[0]: calib_imgs}

    # On-device inference
    device = hub.Device("Samsung Galaxy S23")
    hub_model = compile_zoo_model_to_hub(
        model=model,
        device=device,
        target_runtime=target_runtime,
        calibration_data=calibration_data,
    )

    # Make sure model is quantized
    tgt_model_size_mb = get_model_size_mb(hub_model.model)
    np.testing.assert_allclose(expected_size_mb, tgt_model_size_mb, rtol=0.1)

    # Check on-device accuracy
    hub_out = hub_model(hub_dataset)
    evaluator = model.get_evaluator()
    evaluator.add_batch(hub_out, label_test)
    hub_acc = evaluator.get_accuracy_score()
    print(f"{target_runtime=}, {hub_acc=}")
    np.testing.assert_allclose(expected_acc, hub_acc, atol=0.01)
