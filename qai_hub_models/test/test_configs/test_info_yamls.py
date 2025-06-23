# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.configs.info_yaml import MODEL_USE_CASE, QAIHMModelInfo
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
    "video-object-tracking",
    "other",
}


def test_model_usecase_to_hf_pipeline_tag():
    for use_case in MODEL_USE_CASE:
        assert use_case.map_to_hf_pipeline_tag() in HF_PIPELINE_TAGS


def test_info_yaml():
    for model_id in MODEL_IDS:
        try:
            info_spec = QAIHMModelInfo.from_model(model_id)
            QAIHMModelInfo.model_validate(
                info_spec, context=dict(validate_urls_exist=True)
            )
        except Exception as err:
            assert False, f"{model_id} config validation failed: {str(err)}"

        # Verify model ID is the same as folder name
        assert (
            info_spec.id == model_id
        ), f"{model_id} config ID does not match the model's folder name"
