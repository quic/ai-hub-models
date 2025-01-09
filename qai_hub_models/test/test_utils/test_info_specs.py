# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.configs.info_yaml import (
    MODEL_DOMAIN,
    MODEL_TAG,
    MODEL_USE_CASE,
    QAIHMModelInfo,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

HF_PIPELINE_TAGS = {
    "keypoint-detection",
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    "tabular-to-text",
    "table-to-text",
    "multiple-choice",
    "text-retrieval",
    "time-series-forecasting",
    "text-to-video",
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "graph-ml",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "other",
}


def test_model_usecase_to_hf_pipeline_tag():
    for use_case in MODEL_USE_CASE:
        assert use_case.map_to_hf_pipeline_tag() in HF_PIPELINE_TAGS


def test_info_spec():
    # Guard against MODEL_IDS being empty
    assert (
        len(MODEL_IDS) > 0
    ), "Something went wrong. This test found no models to validate."

    for model_id in MODEL_IDS:
        try:
            info_spec = QAIHMModelInfo.from_model(model_id)
        except Exception as err:
            assert False, f"{model_id} config validation failed: {str(err)}"

        # Verify model ID is the same as folder name
        assert (
            info_spec.id == model_id
        ), f"{model_id} config ID does not match the model's folder name"

        # Validate spec
        reason = info_spec.validate()
        assert reason is None, f"{model_id} config validation failed: {reason}"


def test_qaihm_domain():
    # Test " " is handled correctly and vice-versa
    assert MODEL_DOMAIN.from_string("Computer Vision") == MODEL_DOMAIN.COMPUTER_VISION
    assert MODEL_DOMAIN.COMPUTER_VISION.__str__() == "Computer Vision"


def test_qaihm_tags():
    # Test "-" is handled correctly and vice-versa
    assert MODEL_TAG.from_string("real-time") == MODEL_TAG.REAL_TIME
    assert MODEL_TAG.REAL_TIME.__str__() == "real-time"


def test_qaihm_usecases():
    # Test " " is handled correctly and vice-versa
    assert (
        MODEL_USE_CASE.from_string("Image Classification")
        == MODEL_USE_CASE.IMAGE_CLASSIFICATION
    )
    assert MODEL_USE_CASE.IMAGE_CLASSIFICATION.__str__() == "Image Classification"
