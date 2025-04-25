# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import re
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
    "cc-by-2.0",
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
    "eupl-1.1",
    "agpl-3.0",
    "gfdl",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "isc",
    "intel-research",
    "lppl-1.3c",
    "ms-pl",
    "apple-ascl",
    "apple-amlr",
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
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "gemma",
    "unknown",
    "other",
    "array",
}


@unique
class MODEL_LICENSE(ParseableQAIHMEnum):
    # General
    UNLICENSED = 0  # unlicensed
    AI_HUB_MODELS_LICENSE = 1  # ai-hub-model-license
    COMMERCIAL = 2  # commercial

    # Non-restrictive
    APACHE_2_0 = 30  # apache-2.0
    MIT = 31  # mit
    BSD_3_CLAUSE = 32  # bsd-3-clause
    CC_BY_4_0 = 33  # cc-by-4.0

    # Copyleft (derivative works must have the same license)
    AGPL_3_0 = 52  # agpl-3.0
    GPL_3_0 = 53  # gpl-3.0
    CREATIVEML_OPENRAIL_M = 54  # creativeml-openrail-m

    # Commecial use prohibited
    OTHER_NON_COMMERCIAL = 70  # other-non-commercial
    CC_BY_NON_COMMERCIAL_4_0 = 71  # cc-by-non-commercial-4.0

    # Licenses for specific models
    AIMET_MODEL_ZOO = 90  # aimet-model-zoo
    LLAMA2 = 91  # llama2
    LLAMA3 = 92  # llama3
    TAIDE = 93  # taide

    @property
    def is_copyleft(self) -> bool:
        # Return true if any derivative works must also use this license.
        return self in [
            MODEL_LICENSE.COMMERCIAL,
            MODEL_LICENSE.AGPL_3_0,
            MODEL_LICENSE.GPL_3_0,
            MODEL_LICENSE.CREATIVEML_OPENRAIL_M,
            MODEL_LICENSE.LLAMA2,
            MODEL_LICENSE.LLAMA3,
            MODEL_LICENSE.TAIDE,
        ]

    @property
    def is_non_commerical(self) -> bool:
        # If true if this license does not permit commercial use.
        return self in [
            MODEL_LICENSE.OTHER_NON_COMMERCIAL,
            MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0,
        ]

    @property
    def deploy_license(self) -> MODEL_LICENSE | None:
        # For a given model source license (self), get the associated model deployment license.
        # If None is returned, this model cannot be published with any license.
        if self.is_copyleft:
            return self
        if self.is_non_commerical:
            return None
        return MODEL_LICENSE.AI_HUB_MODELS_LICENSE

    @property
    def huggingface_name(self) -> str:
        hf_str = None
        if self == MODEL_LICENSE.CC_BY_NON_COMMERCIAL_4_0:
            hf_str = "cc-by-nc-4.0"
        elif self == MODEL_LICENSE.UNLICENSED:
            hf_str = "unlicense"
        else:
            hf_str = str(self)
        if hf_str in HF_AVAILABLE_LICENSES:
            return hf_str
        return "other"

    @property
    def url(self) -> str | None:
        # If this license has a known URL, return it.
        if self == MODEL_LICENSE.AI_HUB_MODELS_LICENSE:
            return "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/Qualcomm+AI+Hub+Proprietary+License.pdf"
        elif self in [MODEL_LICENSE.LLAMA2, MODEL_LICENSE.LLAMA3]:
            return "https://github.com/facebookresearch/llama/blob/main/LICENSE"
        elif self == MODEL_LICENSE.TAIDE:
            return "https://en.taide.tw/download.html"
        return None

    @staticmethod
    def from_string(string: str) -> MODEL_LICENSE:
        try:
            return MODEL_LICENSE[string.upper().replace("-", "_").replace(".", "_")]
        except KeyError:
            raise ValueError(
                f"Unknown license type: {string}. See the MODEL_LICENSE enum for valid options (or to add an option)."
            )

    def __str__(self) -> str:
        return re.sub(r"(?<=\d)_(?=\d)", ".", self.name).replace("_", "-").lower()


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
        if self == MODEL_DOMAIN.GENERATIVE_AI:
            return "Generative AI"
        else:
            return self.name.replace("_", " ").title()

    def __repr__(self):
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
    DRIVER_ASSISTANCE = 109

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
        if self.name == "DRIVER_ASSISTANCE":
            return "other"
        return self.name.replace("_", "-").lower()
