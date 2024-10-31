# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run it with pytest --on-device
"""

import numpy as np
import pytest
import qai_hub as hub
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from qai_hub_models.datasets.imagenette import ImagenetteDataset
from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.convnext_tiny_w8a8_quantized.model import (
    ConvNextTinyW8A8Quantizable,
)
from qai_hub_models.models.convnext_tiny_w8a16_quantized.model import (
    ConvNextTinyW8A16Quantizable,
)
from qai_hub_models.models.inception_v3_quantized.model import InceptionNetV3Quantizable
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
def test_data(data_loaders) -> tuple[torch.Tensor, torch.Tensor, hub.Dataset]:
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


@on_device
@pytest.mark.parametrize(
    "model_cls,target_runtime,expected_size_mb,expected_acc",
    [
        (MobileNetV2Quantizable, TargetRuntime.TFLITE, 3.64, 0.816),
        (MobileNetV2Quantizable, TargetRuntime.QNN, 4.09, 0.813),
        (MobileNetV3LargeQuantizable, TargetRuntime.TFLITE, 5.72, 0.848),
        # MobileNetV3LargeQuantizable, TargetRuntime.QNN fails to convert (AISW-87206)
        (ResNet18Quantizable, TargetRuntime.TFLITE, 11.30, 0.805),
        (ResNet18Quantizable, TargetRuntime.QNN, 11.65, 0.796),
        (ResNet50Quantizable, TargetRuntime.TFLITE, 25.09, 0.847),
        (ResNet50Quantizable, TargetRuntime.QNN, 25.41, 0.848),
        (ResNet101Quantizable, TargetRuntime.TFLITE, 43.88, 0.858),
        (ResNet101Quantizable, TargetRuntime.QNN, 44.08, 0.831),
        (ResNeXt50Quantizable, TargetRuntime.TFLITE, 24.77, 0.891),
        (ResNeXt50Quantizable, TargetRuntime.QNN, 25.03, 0.893),
        (ResNeXt101Quantizable, TargetRuntime.TFLITE, 87.28, 0.926),
        # Fails to infer (#9827)
        (ResNeXt101Quantizable, TargetRuntime.QNN, 87.26, None),
        (SqueezeNetQuantizable, TargetRuntime.TFLITE, 1.30, 0.637),
        (SqueezeNetQuantizable, TargetRuntime.QNN, 1.69, 0.636),
        (RegNetQuantizable, TargetRuntime.TFLITE, 15.42, 0.872),
        (RegNetQuantizable, TargetRuntime.QNN, 15.89, 0.876),
        (WideResNet50Quantizable, TargetRuntime.TFLITE, 66.59, 0.923),
        (WideResNet50Quantizable, TargetRuntime.QNN, 66.86, 0.922),
        (ShufflenetV2Quantizable, TargetRuntime.TFLITE, 1.46, 0.674),
        (ShufflenetV2Quantizable, TargetRuntime.QNN, 1.99, 0.670),
        (InceptionNetV3Quantizable, TargetRuntime.TFLITE, 23.32, 0.841),
        (InceptionNetV3Quantizable, TargetRuntime.QNN, 23.85, 0.845),
        # ConvNextTinyW8A8Quantizable, SourceModelFormat.ONNX not supported yet (#10862)
        (ConvNextTinyW8A8Quantizable, TargetRuntime.QNN, 28.33, 0.846),
        # ConvNextTinyW8A16Quantizable, SourceModelFormat.ONNX not supported yet (#10862)
        (ConvNextTinyW8A16Quantizable, TargetRuntime.QNN, 28.34, 0.876),
    ],
)
def test_quantized_accuracy(
    model_cls,
    target_runtime,
    expected_size_mb,
    expected_acc,
    test_data,
    data_loaders,
):
    """
    1. Export and compile quantized_model on Hub.
    2. Run inference on Hub on test.

    Note: We don't run profile job to get perf here but leave that to the score card.
    """
    model = model_cls.from_pretrained()

    img_test, label_test, hub_dataset = test_data
    calib_loader, test_loader = data_loaders

    calibration_data = model.get_calibration_data(target_runtime)

    # On-device inference
    device = hub.Device("Samsung Galaxy S23")
    hub_model = compile_zoo_model_to_hub(
        model=model,
        source_model_format=SourceModelFormat.ONNX,
        device=device,
        target_runtime=target_runtime,
        calibration_data=calibration_data,
    )

    # Make sure model is quantized
    tgt_model_size_mb = get_model_size_mb(hub_model.model)
    print(f"{model_cls=}, {target_runtime=}, {tgt_model_size_mb=}")
    np.testing.assert_allclose(expected_size_mb, tgt_model_size_mb, rtol=0.1)

    if expected_acc is None:
        pytest.skip("Fails to infer")

    # Check on-device accuracy
    hub_out = hub_model(hub_dataset)
    evaluator = model.get_evaluator()
    evaluator.add_batch(hub_out, label_test)
    hub_acc = evaluator.get_accuracy_score()
    print(f"{model_cls=}, {target_runtime=}, {hub_acc=}")
    np.testing.assert_allclose(expected_acc, hub_acc, atol=0.01)
