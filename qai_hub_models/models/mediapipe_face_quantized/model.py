# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
    constrain_quantized_inputs_to_image_range,
    tie_observers,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v1.quantsim import QuantizationSimModel

from qai_hub_models.models.mediapipe_face.model import (
    FaceDetector,
    FaceLandmarkDetector,
    MediaPipeFace,
)
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
DEFAULT_LANDMARK_DETECTOR_ENCODINGS = "landmark_detector_quantized_encodings.json"

# This encodings file contains manual overrides.
# The final layers output box confidence logits, most of which are extremely negative.
# There are also only a few values that will be positive, which are the ones we care about.
# When calibrating the range, the quantizer clips the range to exclude these positive outliers.
# The range was manually overridden in the encodings file to be [-128, 127]
DEFAULT_FACE_DETECTOR_ENCODINGS = "face_detector_quantized_encodings.json"


class MediaPipeFaceQuantizable(MediaPipeFace):
    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        face_detector_encodings: str | None = "DEFAULT",
        landmark_detector_encodings: str | None = "DEFAULT",
    ) -> MediaPipeFaceQuantizable:
        return cls(
            FaceDetectorQuantizable.from_pretrained(face_detector_encodings),
            FaceLandmarkDetectorQuantizable.from_pretrained(
                landmark_detector_encodings
            ),
        )


class FaceDetectorQuantizable(AIMETQuantizableMixin, FaceDetector):
    def __init__(self, detector: QuantizationSimModel, anchors: torch.Tensor):
        FaceDetector.__init__(self, detector.model, anchors)
        AIMETQuantizableMixin.__init__(self, detector)

    @classmethod
    def from_pretrained(cls, aimet_encodings: str | None = "DEFAULT"):  # type: ignore[override]
        fp16_model = FaceDetector.from_pretrained()
        input_shape = cls.get_input_spec()["image"][0]
        model = prepare_model(fp16_model)
        equalize_model(model, input_shape)
        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        tie_observers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        final_model = cls(sim, fp16_model.anchors)

        if aimet_encodings:
            aimet_encodings_to_load = (
                aimet_encodings
                if aimet_encodings != "DEFAULT"
                else CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_FACE_DETECTOR_ENCODINGS
                ).fetch()
            )

            final_model.load_encodings(
                str(aimet_encodings_to_load),
                strict=False,
                partial=False,
                requires_grad=None,
                allow_overwrite=None,
            )

        return final_model


class FaceLandmarkDetectorQuantizable(AIMETQuantizableMixin, FaceLandmarkDetector):
    def __init__(self, detector: QuantizationSimModel):
        FaceLandmarkDetector.__init__(self, detector.model)
        AIMETQuantizableMixin.__init__(self, detector)

    @classmethod
    def from_pretrained(cls, aimet_encodings: str | None = "DEFAULT"):  # type: ignore[override]
        fp16_model = FaceLandmarkDetector.from_pretrained()
        input_shape = cls.get_input_spec()["image"][0]
        model = prepare_model(fp16_model)
        equalize_model(model, input_shape)
        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        tie_observers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        final_model = cls(sim)

        if aimet_encodings:
            aimet_encodings_to_load = (
                aimet_encodings
                if aimet_encodings != "DEFAULT"
                else CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_LANDMARK_DETECTOR_ENCODINGS
                ).fetch()
            )

            final_model.load_encodings(
                str(aimet_encodings_to_load),
                strict=False,
                partial=False,
                requires_grad=None,
                allow_overwrite=None,
            )

        return final_model
