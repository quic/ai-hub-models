# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import unique

from qai_hub_models.utils.base_config import ParseableQAIHMEnum

HF_AVAILABLE_LICENSES = {
    "apache-2.0",
    "mit",
    "openrail",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "bigcode-openrail-m",
    "afl-3.0",
    "artistic-2.0",
    "bsl-1.0",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "c-uda",
    "cc",
    "cc0-1.0",
    "cc0-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-4.0",
    "cdla-sharing-1.0",
    "cdla-permissive-1.0",
    "cdla-permissive-2.0",
    "wtfpl",
    "ecl-2.0",
    "epl-1.0",
    "epl-2.0",
    "etalab-2.0",
    "agpl-3.0",
    "gfdl",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "isc",
    "lppl-1.3c",
    "ms-pl",
    "mpl-2.0",
    "odc-by",
    "odbl",
    "openrail++",
    "osl-3.0",
    "postgresql",
    "ofl-1.1",
    "ncsa",
    "unlicense",
    "zlib",
    "pddl",
    "lgpl-lr",
    "deepfloyd-if-license",
    "llama2",
    "llama3",
    "unknown",
    "other",
}


@unique
class MODEL_DOMAIN(ParseableQAIHMEnum):
    COMPUTER_VISION = 0  # YAML string: Computer Vision
    AUDIO = 1  # YAML string: Auto
    MULTIMODAL = 2  # YAML string: Multimodel
    GENERATIVE_AI = 3  # YAML string: Generative AI

    @staticmethod
    def from_string(string: str) -> MODEL_DOMAIN:
        # It's quicker to get the domain first via hashmap then verify the string against it.
        domain = MODEL_DOMAIN[string.upper().replace(" ", "_")]

        # Each string is compared becuase the YAML needs specific capitalization.
        # The public website takes these strings directly from the YAML and displays them.
        if domain == MODEL_DOMAIN.GENERATIVE_AI:
            assert (
                string == "Generative AI"
            ), f"Expected 'Generative AI' for Model Domain, but capitalization is wrong: {string}"
        else:
            assert (
                string == domain.name.replace("_", " ").title()
            ), f"Model Domain should be in Title Case ({string.title()}, but got {string} instead."

        return domain

    def __str__(self):
        return self.name.title().replace("_", " ")


@unique
class MODEL_TAG(ParseableQAIHMEnum):
    BACKBONE = 0  # YAML string: backbone
    REAL_TIME = 1  # YAML string: real-time
    FOUNDATION = 2  # YAML string: foundation
    QUANTIZED = 3  # YAML string: quantized
    LLM = 4  # YAML string: llm
    GENERATIVE_AI = 5  # YAML string: generative-ai

    @staticmethod
    def from_string(string: str) -> MODEL_TAG:
        # YAML should contain lower case strings that are dash separated.
        assert (
            "_" not in string and " " not in string
        ), f"Multi-Word Model Tags should be split by '-', not ' ' or '_': {string}"
        assert string == string.lower(), f"Model Tags should be lower case: {string}"
        return MODEL_TAG[string.upper().replace("-", "_")]

    def __str__(self) -> str:
        return self.name.replace("_", "-").lower()

    def __repr__(self) -> str:
        return self.__str__()


@unique
class MODEL_STATUS(ParseableQAIHMEnum):
    PUBLIC = 0
    PRIVATE = 1
    # proprietary models are released only internally
    PROPRIETARY = 2

    @staticmethod
    def from_string(string: str) -> MODEL_STATUS:
        return MODEL_STATUS[string.upper()]

    def __str__(self):
        return self.name


@unique
class MODEL_USE_CASE(ParseableQAIHMEnum):
    # Image: 100 - 199
    IMAGE_CLASSIFICATION = 100
    IMAGE_EDITING = 101
    IMAGE_GENERATION = 102
    SUPER_RESOLUTION = 103
    SEMANTIC_SEGMENTATION = 104
    DEPTH_ESTIMATION = 105
    # Ex: OCR, image caption
    IMAGE_TO_TEXT = 106
    OBJECT_DETECTION = 107
    POSE_ESTIMATION = 108

    # Audio: 200 - 299
    SPEECH_RECOGNITION = 200
    AUDIO_ENHANCEMENT = 201
    AUDIO_CLASSIFICATION = 202

    # Video: 300 - 399
    VIDEO_CLASSIFICATION = 300
    VIDEO_GENERATION = 301

    # LLM: 400 - 499
    TEXT_GENERATION = 400

    @staticmethod
    def from_string(string: str) -> MODEL_USE_CASE:
        # It's quicker to get the use case first via hashmap then verify the string against it.
        use_case = MODEL_USE_CASE[string.upper().replace(" ", "_")]

        # Each string is compared becuase the YAML needs specific capitalization.
        # The public website takes these strings directly from the YAML and displays them.
        assert (
            string == use_case.name.replace("_", " ").title()
        ), f"Use Case should be in Title Case: {string}"

        return use_case

    def __str__(self):
        return self.name.replace("_", " ").title()

    def map_to_hf_pipeline_tag(self):
        """Map our usecase to pipeline-tag used by huggingface."""
        if self.name in {"IMAGE_EDITING", "SUPER_RESOLUTION"}:
            return "image-to-image"
        if self.name == "SEMANTIC_SEGMENTATION":
            return "image-segmentation"
        if self.name == "POSE_ESTIMATION":
            return "keypoint-detection"
        if self.name == "AUDIO_ENHANCEMENT":
            return "audio-to-audio"
        if self.name == "VIDEO_GENERATION":
            return "image-to-video"
        if self.name == "IMAGE_GENERATION":
            return "unconditional-image-generation"
        if self.name == "SPEECH_RECOGNITION":
            return "automatic-speech-recognition"
        if self.name == "AUDIO_CLASSIFICATION":
            return "audio-classification"
        return self.name.replace("_", "-").lower()
