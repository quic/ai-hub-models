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
from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.mobilenet_v2_quantized.model import MobileNetV2Quantizable
from qai_hub_models.models.mobilenet_v3_large_quantized.model import (
    MobileNetV3LargeQuantizable,
)
from qai_hub_models.models.regnet_quantized.model import RegNetQuantizable
from qai_hub_models.models.resnet18_quantized.model import ResNet18Quantizable
from qai_hub_models.models.resnet50_quantized.model import ResNet50Quantizable
from qai_hub_models.models.resnet101_quantized.model import ResNet101Quantizable
from qai_hub_models.models.resnext50_quantized.model import ResNeXt50Quantizable
from qai_hub_models.models.resnext101_quantized.model import ResNeXt101Quantizable
from qai_hub_models.models.shufflenet_v2_quantized.model import ShufflenetV2Quantizable
from qai_hub_models.models.squeezenet1_1_quantized.model import SqueezeNetQuantizable
from qai_hub_models.models.wideresnet50_quantized.model import WideResNet50Quantizable
from qai_hub_models.utils.base_model import SourceModelFormat, TargetRuntime
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
    input_name = list(ImagenetClassifier.get_input_spec().keys())[0]
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


@pytest.fixture(
    scope="module",
    params=[
        # Class, AIMET accuracy
        (MobileNetV2Quantizable, 0.8100),
        (MobileNetV3LargeQuantizable, 0.8550),
        (ResNet18Quantizable, 0.8010),
        (ResNet50Quantizable, 0.8520),
        (ResNet101Quantizable, 0.8530),
        (ResNeXt50Quantizable, 0.8880),
        (ResNeXt101Quantizable, 0.9250),
        (SqueezeNetQuantizable, 0.6410),
        (RegNetQuantizable, 0.8750),
        (WideResNet50Quantizable, 0.9190),
        (ShufflenetV2Quantizable, 0.6740),
    ],
)
def quantized_model(request, data_loaders, test_data):
    """
    Create encoding from calibration data and returned quantized model with
    validated off-target accuracy computed on QuantSim
    """
    img_test, label_test, hub_dataset = test_data
    calib_loader, test_loader = data_loaders
    model_cls, target_sim_acc = request.param
    model = model_cls.from_pretrained(aimet_encodings=None)

    # Calibration in quantization
    num_calib_batches = 3
    model.quantize(calib_loader, num_calib_batches, data_has_gt=True)

    # QuantSim evaluation on eval set
    evaluator = model.get_evaluator()

    batch_size = 32
    for i in tqdm(list(range(0, img_test.size(0), batch_size)), desc="QuantSim eval"):
        img_batch = img_test[i : i + batch_size]
        label_batch = label_test[i : i + batch_size]

        sim_out = model(img_batch).detach()
        evaluator.add_batch(sim_out, label_batch)

    sim_acc = evaluator.get_accuracy_score()
    print(f"{model_cls=}, {sim_acc=}")
    np.testing.assert_allclose(target_sim_acc, sim_acc, atol=0.01)
    return model


@on_device
@pytest.mark.parametrize(
    "source_model_format,target_runtime,hub_needs_calib_data",
    [
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, False),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, False),
    ],
)
def test_make_encoding_w8a8_accuracy(
    source_model_format,
    target_runtime,
    hub_needs_calib_data,
    test_data,
    quantized_model,
    data_loaders,
):
    """
    1. Export and compile quantized_model on Hub.
    2. Run inference on Hub on test.

    Note: We don't run profile job to get perf here but leave that to the score card.
    """
    model = quantized_model

    expected_size_mb_and_acc = {
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, MobileNetV2Quantizable): (
            3.64,
            0.801,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, MobileNetV2Quantizable): (
            4.02,
            0.801,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, MobileNetV3LargeQuantizable): (
            5.79,
            0.859,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, MobileNetV3LargeQuantizable): (
            None,  # Fails to convert (AISW-87206)
            None,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ResNet18Quantizable): (
            11.30,
            0.778,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ResNet18Quantizable): (
            11.61,
            0.789,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ResNet50Quantizable): (
            25.09,
            0.837,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ResNet50Quantizable): (
            25.33,
            0.834,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ResNet101Quantizable): (
            43.89,
            0.827,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ResNet101Quantizable): (
            44.08,
            0.831,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ResNeXt50Quantizable): (
            24.77,
            0.888,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ResNeXt50Quantizable): (
            24.96,
            0.888,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ResNeXt101Quantizable): (
            87.29,
            0.906,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ResNeXt101Quantizable): (
            87.11,
            None,  # Fails to infer (#9827)
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, SqueezeNetQuantizable): (
            1.30,
            0.609,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, SqueezeNetQuantizable): (
            1.66,
            0.609,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, RegNetQuantizable): (
            15.43,
            0.859,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, RegNetQuantizable): (
            15.77,
            0.859,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, WideResNet50Quantizable): (
            66.59,
            0.900,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, WideResNet50Quantizable): (
            66.78,
            0.897,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, ShufflenetV2Quantizable): (
            1.47,
            0.661,
        ),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, ShufflenetV2Quantizable): (
            1.90,
            0.661,
        ),
    }
    expected_size_mb, expected_acc = expected_size_mb_and_acc[
        (source_model_format, target_runtime, model.__class__)
    ]
    if expected_size_mb is None:
        pytest.skip("Fails to compile")

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
        source_model_format=source_model_format,
        device=device,
        target_runtime=target_runtime,
        calibration_data=calibration_data,
    )

    # Make sure model is quantized
    tgt_model_size_mb = get_model_size_mb(hub_model.model)
    model_cls = quantized_model.__class__
    print(
        f"{model_cls=}, {source_model_format=}, {target_runtime=}, {tgt_model_size_mb=}"
    )
    np.testing.assert_allclose(expected_size_mb, tgt_model_size_mb, rtol=0.1)

    if expected_acc is None:
        pytest.skip("Fails to infer")

    # Check on-device accuracy
    hub_out = hub_model(hub_dataset)
    evaluator = model.get_evaluator()
    evaluator.add_batch(hub_out, label_test)
    hub_acc = evaluator.get_accuracy_score()
    print(f"{model_cls=}, {source_model_format=}, {target_runtime=}, {hub_acc=}")
    np.testing.assert_allclose(expected_acc, hub_acc, atol=0.01)
