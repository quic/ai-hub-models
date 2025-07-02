# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import inspect
from typing import cast

from qai_hub_models.utils.input_spec import InputSpec

from .common import BaseDataset, DatasetSplit

_ALL_DATASETS_IMPORT_ERRORS: dict[str, Exception] = {}
DATASET_NAME_MAP: dict[str, type[BaseDataset]] = {}


# We don't want to require a user to install requirements for all datasets just to
# import the datasets folder. Therefore we only include the datasets that can
# be imported.
def _try_import_dataset(module_name: str, cls: str):
    """
    Import the dataset and add it to the DATASET_NAME_MAP, or pass
    if dependencies for the dataset aren't installed.
    """
    try:
        module = importlib.import_module(module_name, package="qai_hub_models.datasets")
    except ModuleNotFoundError as e:
        if module_name.startswith("."):
            module_name = module_name[1:]
        if str(e) == f"No module named 'qai_hub_models.datasets.{module_name}":
            # this module legitimately does not exist
            raise e

        # The module couldn't be loaded for some other reason.
        #
        # By default, the name of the dataset is the name of its module.
        # We add it to this import errors list to hopefully raise the error
        # at a later time (when the user requests this dataset).
        _ALL_DATASETS_IMPORT_ERRORS[module_name] = e
        return
    except NotImplementedError as e:
        if "AIMET-ONNX" in str(e):
            # stable diffusion dataset requires aimet-onnx
            _ALL_DATASETS_IMPORT_ERRORS[module_name] = e
            return
        raise e

    if x := getattr(module, cls, None):
        xds = cast(type[BaseDataset], x)
        DATASET_NAME_MAP[xds.dataset_name()] = xds
    else:
        raise ValueError(
            f"Could not import {cls}. {cls} was not found in {module_name}"
        )


_try_import_dataset(".bsd300", "BSD300Dataset")
_try_import_dataset(".cityscapes", "CityscapesDataset")
_try_import_dataset(".cityscapes_lowres", "CityscapesLowResDataset")
_try_import_dataset(".ade20k", "ADESegmentationDataset")
_try_import_dataset(".coco", "CocoDataset")
_try_import_dataset(".coco_seg", "CocoSegDataset")
_try_import_dataset(".ppe", "PPEDataset")
_try_import_dataset(".coco91class", "Coco91ClassDataset")
_try_import_dataset(".coco_face", "CocoFaceDataset")
_try_import_dataset(".human_faces", "HumanFacesDataset")
_try_import_dataset(".human_faces", "HumanFaces192Dataset")
_try_import_dataset(".coco_panoptic_seg", "CocoPanopticSegmentationDataset")
_try_import_dataset(".foot_track_dataset", "FootTrackDataset")
_try_import_dataset(".gear_guard_dataset", "GearGuardDataset")
_try_import_dataset(".face_attrib_dataset", "FaceAttribDataset")
_try_import_dataset(".face_det_lite", "FaceDetLiteDataset")
_try_import_dataset(".coco_foot_track_dataset", "CocoFootTrackDataset")
_try_import_dataset(".kinetics400", "Kinetics400Dataset")
_try_import_dataset(".kinetics400_224", "Kinetics400_224Dataset")
_try_import_dataset(".imagenet", "ImagenetDataset")
_try_import_dataset(".imagenet_colorization", "ImagenetColorizationDataset")
_try_import_dataset(".imagenet_256", "Imagenet_256Dataset")
_try_import_dataset(".imagenette", "ImagenetteDataset")
_try_import_dataset(".imagenette_colorization", "ImagenetteColorizationDataset")
_try_import_dataset(".imagenette_256", "Imagenette_256Dataset")
_try_import_dataset(".nyuv2", "NyUv2Dataset")
_try_import_dataset(".nyuv2x518", "NyUv2x518Dataset")
_try_import_dataset(".pascal_voc", "VOCSegmentationDataset")
_try_import_dataset(".mpii", "MPIIDataset")
_try_import_dataset(".cocobody", "CocoBodyDataset")
_try_import_dataset(".cocowholebody", "CocoWholeBodyDataset")
_try_import_dataset(".cocobody_513x257", "CocoBody513x257Dataset")
_try_import_dataset(".coco_face_480x640", "CocoFace_480x640Dataset")
_try_import_dataset(".carvana", "CarvanaDataset")
_try_import_dataset(".camouflage_dataset", "CamouflageDataset")
_try_import_dataset(".eg1800", "eg1800SegmentationDataset")
_try_import_dataset(".stable_diffusion_calib", "StableDiffusionCalibDatasetTextEncoder")
_try_import_dataset(".stable_diffusion_calib", "StableDiffusionCalibDatasetUnet")
_try_import_dataset(".stable_diffusion_calib", "StableDiffusionCalibDatasetVae")
_try_import_dataset(".celebahq", "CelebAHQDataset")


def get_dataset_from_name(
    name: str, split: DatasetSplit, input_spec: InputSpec | None = None, **kwargs
) -> BaseDataset:
    dataset_cls = DATASET_NAME_MAP.get(name, None)
    if not dataset_cls:
        if name in _ALL_DATASETS_IMPORT_ERRORS:
            raise _ALL_DATASETS_IMPORT_ERRORS[name]
        raise ValueError(f"Unable to find dataset with name {name}")

    # Some datasets can be configured to different input specs
    if (
        input_spec is not None
        and "input_spec" in inspect.signature(dataset_cls.__init__).parameters
    ):
        kwargs["input_spec"] = input_spec
    return dataset_cls(split=split, **kwargs)  # type: ignore[call-arg]
