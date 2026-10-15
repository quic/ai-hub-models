# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.deeplabv3_resnet50.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.deeplabv3_resnet50.demo import main as demo_main
from qai_hub_models.models.deeplabv3_resnet50.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DeepLabV3_ResNet50,
)
from qai_hub_models.models.templates.deeplab.app import DeepLabV3App
from qai_hub_models.models.templates.deeplab.model import NUM_CLASSES
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

OUTPUT_IMAGE_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "deeplab_output_mask.npy"
)


def test_task() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    app = DeepLabV3App(DeepLabV3_ResNet50.from_pretrained(), num_classes=NUM_CLASSES)
    output_mask = app.predict(image, True)
    output_mask_gt = load_numpy(OUTPUT_IMAGE_MASK)
    assert (output_mask == output_mask_gt).mean() > 0.95


def test_demo() -> None:
    demo_main(is_test=True)
