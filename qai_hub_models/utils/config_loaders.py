# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum, unique
from pathlib import Path
from types import GenericAlias
from typing import Any, Optional, TypeVar, Union, get_args, get_type_hints

import requests
from qai_hub.util.session import create_session
from schema import And
from schema import Optional as OptionalSchema
from schema import Schema

from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET, load_yaml
from qai_hub_models.utils.path_helpers import (
    MODELS_PACKAGE_NAME,
    QAIHM_PACKAGE_NAME,
    get_qaihm_models_root,
    get_qaihm_package_root,
)

QAIHM_PACKAGE_ROOT = get_qaihm_package_root()
QAIHM_MODELS_ROOT = get_qaihm_models_root()
QAIHM_DIRS = [
    Path(f.path)
    for f in os.scandir(QAIHM_MODELS_ROOT)
    if f.is_dir() and "info.yaml" in os.listdir(f)
]
MODEL_IDS = [f.name for f in QAIHM_DIRS]

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


def _get_origin(input_type: type) -> type:
    """
    For nested types like list[str] or Union[str, int], this function will
        return the "parent" type like List or Union.

    If the input type is not a nested type, the function returns the input_type.
    """
    return getattr(input_type, "__origin__", input_type)


def _extract_optional_type(input_type: type) -> type:
    """
    Given an optional type as input, returns the inner type that is wrapped.

    For example, if input type is Optional[int], the function returns int.
    """
    assert (
        _get_origin(input_type) == Union
    ), "Input type must be an instance of `Optional`."
    union_args = get_args(input_type)
    assert len(union_args) == 2 and issubclass(
        union_args[1], type(None)
    ), "Input type must be an instance of `Optional`."
    return union_args[0]


def _constructor_from_type(input_type: type) -> Union[type, Callable]:
    """
    Given a type, return the appropriate constructor for that type.

    For primitive types like str and int, the type and constructor are the same object.

    For types like List, the constructor is list.
    """
    if input_type == GenericAlias:
        return input_type
    input_type = _get_origin(input_type)
    if input_type == list:
        return list
    if input_type == dict:
        return dict
    return input_type


@dataclass
class BaseDataClass:
    @classmethod
    def get_schema(cls) -> Schema:
        """Derive the Schema from the fields set on the dataclass."""
        schema_dict = {}
        field_datatypes = get_type_hints(cls)
        for field in fields(cls):
            field_type = field_datatypes[field.name]
            if _get_origin(field_type) == Union:
                field_type = _extract_optional_type(field_type)
                assert (
                    field.default != dataclasses.MISSING
                ), "Optional fields must have a default set."
            if field.default != dataclasses.MISSING:
                field_key = OptionalSchema(field.name, default=field.default)
            elif field.default_factory != dataclasses.MISSING:
                field_key = OptionalSchema(field.name, default=field.default_factory())
            else:
                field_key = field.name
            schema_dict[field_key] = _constructor_from_type(field_type)
        return Schema(And(schema_dict))

    @classmethod
    def from_dict(
        cls: type[BaseDataClassTypeVar], val_dict: dict[str, Any]
    ) -> BaseDataClassTypeVar:
        # Validate schema
        val_dict = cls.get_schema().validate(val_dict)

        # Build dataclass
        kwargs = {field.name: val_dict[field.name] for field in fields(cls)}
        return cls(**kwargs)

    def validate(self) -> Optional[str]:
        """Returns a string reason if the this class is not valid, or None if it is valid"""
        return None

    @classmethod
    def from_yaml(
        cls: type[BaseDataClassTypeVar], path: str | Path
    ) -> BaseDataClassTypeVar:
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


BaseDataClassTypeVar = TypeVar("BaseDataClassTypeVar", bound="BaseDataClass")


@unique
class MODEL_DOMAIN(Enum):
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
class MODEL_TAG(Enum):
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


def is_gen_ai_model(tags: list[MODEL_TAG]) -> bool:
    return MODEL_TAG.LLM in tags or MODEL_TAG.GENERATIVE_AI in tags


@unique
class MODEL_STATUS(Enum):
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
class MODEL_USE_CASE(Enum):
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
        return self.name.replace("_", "-").lower()


TFLITE_PATH = "torchscript_onnx_tflite"
QNN_PATH = "torchscript_onnx_qnn"


def bytes_to_mb(num_bytes: int) -> int:
    return round(num_bytes / (1 << 20))


class QAIHMModelPerf:
    """Class to read the perf.yaml and parse it for displaying it on HuggingFace."""

    ###
    # Helper Struct Classes
    ###

    @dataclass
    class PerformanceDetails:
        job_id: str
        inference_time_microsecs: float
        peak_memory_bytes: tuple[int, int]  # min, max
        compute_unit_counts: dict[str, int]
        primary_compute_unit: str
        precision: str

        @staticmethod
        def from_dict(device_perf_details: dict) -> QAIHMModelPerf.PerformanceDetails:
            peak_memory = device_perf_details["estimated_peak_memory_range"]
            layer_info = device_perf_details["layer_info"]
            compute_unit_counts = {}
            for layer_name, count in layer_info.items():
                if "layers_on" in layer_name:
                    if count > 0:
                        compute_unit_counts[layer_name[-3:].upper()] = count

            return QAIHMModelPerf.PerformanceDetails(
                job_id=device_perf_details["job_id"],
                inference_time_microsecs=float(device_perf_details["inference_time"]),
                peak_memory_bytes=(peak_memory["min"], peak_memory["max"]),
                compute_unit_counts=compute_unit_counts,
                primary_compute_unit=device_perf_details["primary_compute_unit"],
                precision=device_perf_details["precision"],
            )

    @dataclass
    class LLMPerformanceDetails:
        time_to_first_token_range_secs: tuple[str, str]  # min, max
        tokens_per_second: float

        @staticmethod
        def from_dict(
            device_perf_details: dict,
        ) -> QAIHMModelPerf.LLMPerformanceDetails:
            ttftr = device_perf_details["time_to_first_token_range"]
            return QAIHMModelPerf.LLMPerformanceDetails(
                time_to_first_token_range_secs=(
                    # Original data is in microseconds
                    str(float(ttftr["min"]) * 1e-6),
                    str(float(ttftr["max"]) * 1e-6),
                ),
                tokens_per_second=device_perf_details["tokens_per_second"],
            )

    @dataclass
    class EvaluationDetails(BaseDataClass):
        name: str
        value: float
        unit: str

    @dataclass
    class DeviceDetails(BaseDataClass):
        name: str
        os: str
        form_factor: str
        os_name: str
        manufacturer: str
        chipset: str

    @dataclass
    class ProfilePerfDetails:
        path: ScorecardProfilePath
        perf_details: QAIHMModelPerf.PerformanceDetails | QAIHMModelPerf.LLMPerformanceDetails
        eval_details: Optional[QAIHMModelPerf.EvaluationDetails] = None

        @staticmethod
        def from_dict(
            path: ScorecardProfilePath, perf_details_dict: dict
        ) -> QAIHMModelPerf.ProfilePerfDetails:
            perf_details: QAIHMModelPerf.LLMPerformanceDetails | QAIHMModelPerf.PerformanceDetails
            if llm_metrics := perf_details_dict.get("llm_metrics", None):
                perf_details = QAIHMModelPerf.LLMPerformanceDetails.from_dict(
                    llm_metrics
                )
            else:
                perf_details = QAIHMModelPerf.PerformanceDetails.from_dict(
                    perf_details_dict
                )

            if eval_metrics := perf_details_dict.get("evaluation_metrics", None):
                eval_details_data = (
                    QAIHMModelPerf.EvaluationDetails.get_schema().validate(eval_metrics)
                )
                eval_details = QAIHMModelPerf.EvaluationDetails.from_dict(
                    eval_details_data
                )
            else:
                eval_details = None

            return QAIHMModelPerf.ProfilePerfDetails(
                path=path, perf_details=perf_details, eval_details=eval_details
            )

    @dataclass
    class DevicePerfDetails:
        device: QAIHMModelPerf.DeviceDetails
        details_per_path: dict[ScorecardProfilePath, QAIHMModelPerf.ProfilePerfDetails]

        @staticmethod
        def from_dict(
            device: QAIHMModelPerf.DeviceDetails, device_runtime_details: dict
        ) -> QAIHMModelPerf.DevicePerfDetails:
            details_per_path = {}
            for profile_path in ScorecardProfilePath:
                if profile_path.long_name in device_runtime_details:
                    perf_details_dict = device_runtime_details[profile_path.long_name]
                    details_per_path[
                        profile_path
                    ] = QAIHMModelPerf.ProfilePerfDetails.from_dict(
                        profile_path, perf_details_dict
                    )
            return QAIHMModelPerf.DevicePerfDetails(
                device=device, details_per_path=details_per_path
            )

    @dataclass
    class ModelPerfDetails:
        model: str
        details_per_device: dict[str, QAIHMModelPerf.DevicePerfDetails]

        @staticmethod
        def from_dict(
            model: str, model_performance_metrics: list[dict]
        ) -> QAIHMModelPerf.ModelPerfDetails:
            details_per_device = {}
            for device_perf_details in model_performance_metrics:
                device_details_data = (
                    QAIHMModelPerf.DeviceDetails.get_schema().validate(
                        device_perf_details["reference_device_info"]
                    )
                )
                device_details = QAIHMModelPerf.DeviceDetails.from_dict(
                    device_details_data
                )
                details_per_device[
                    device_details.name
                ] = QAIHMModelPerf.DevicePerfDetails.from_dict(
                    device_details, device_perf_details
                )

            return QAIHMModelPerf.ModelPerfDetails(
                model=model, details_per_device=details_per_device
            )

    def __init__(self, perf_yaml_path: str | Path, model_name: str):
        self.model_name = model_name
        self.perf_yaml_path = perf_yaml_path
        self.per_model_details: dict[str, QAIHMModelPerf.ModelPerfDetails] = {}

        if os.path.exists(self.perf_yaml_path):
            self.perf_details = load_yaml(self.perf_yaml_path)
            all_models_and_perf = self.perf_details["models"]
            if not isinstance(all_models_and_perf, list):
                all_models_and_perf = [all_models_and_perf]

            for model_perf in all_models_and_perf:
                model_name = model_perf["name"]
                self.per_model_details[
                    model_name
                ] = QAIHMModelPerf.ModelPerfDetails.from_dict(
                    model_name, model_perf["performance_metrics"]
                )


@dataclass
class QAIHMModelCodeGen(BaseDataClass):
    # Whether the model is quantized with aimet.
    is_aimet: bool = False

    # aimet model can additionally specify num calibration samples to speed up
    # compilation
    num_calibration_samples: Optional[int] = None

    # Whether the model's demo supports running on device with the `--on-device` flag.
    has_on_target_demo: bool = False

    # Should print a statement at the end of export script to point to genie tutorial or not.
    add_genie_url_to_export: bool = False

    # If the model doesn't work on qnn, this should explain why,
    # ideally with a reference to an internal issue.
    qnn_export_failure_reason: str = ""

    # If the model doesn't work on tflite, this should explain why,
    # ideally with a reference to an internal issue.
    tflite_export_failure_reason: str = ""

    # If the model doesn't work on onnx, this should explain why,
    # ideally with a reference to an internal issue.
    onnx_export_failure_reason: str = ""

    # If set, changes the default device when running export.py for the model.
    default_device: Optional[str] = None

    # Sets the `check_trace` argument on `torch.jit.trace`.
    check_trace: bool = True

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
    outputs_to_skip_validation: Optional[list[str]] = None

    # Additional arguments to initialize the model when unit testing export.
    # This is commonly used to test a smaller variant in the unit test.
    export_test_model_kwargs: Optional[dict[str, str]] = None

    # Some models are comprised of submodels that should be compiled separately.
    # For example, this is used when there is an encoder/decoder pattern.
    # This is a dict from component name to a python expression that can be evaluated
    # to produce the submodel. The expression can assume the parent model has been
    # initialized and assigned to the variable `model`.
    components: Optional[dict[str, Any]] = None

    # If components is set, this field can specify a subset of components to run
    # by default when invoking `export.py`. If unset, all components are run by default.
    default_components: Optional[list[str]] = None

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Whether the model uses the pre-compiled pattern instead of the
    # standard pre-trained pattern.
    is_precompiled: bool = False

    # If set, disables generating `export.py`.
    skip_export: bool = False

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    # A list of optimizations from `torch.utils.mobile_optimizer` that will
    # speed up the conversion to torchscript.
    torchscript_opt: Optional[list[str]] = None

    # A comma separated list of metrics to print in the inference summary of `export.py`.
    inference_metrics: str = "psnr"

    # Additional details that can be set on the model's readme.
    additional_readme_section: str = ""

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # If set, generates an `evaluate.py` file which can be used to evaluate the model
    # on a full dataset. Datasets specified here must be chosen from `qai_hub_models/datasets`.
    eval_datasets: Optional[list[str]] = None

    # If set, quantizes the model using AI Hub quantize job. This also requires setting
    # the `eval_datasets` field. Calibration data will be pulled from the first item
    # in `eval_datasets`.
    use_hub_quantization: bool = False

    # By default inference tests are done using 8gen1 chipset to avoid overloading
    # newer devices. Some models don't work on 8gen1, so use 8gen3 for those.
    inference_on_8gen3: bool = False

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: Optional[str] = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: Optional[str] = None

    @classmethod
    def from_model(cls: type[QAIHMModelCodeGen], model_id: str) -> QAIHMModelCodeGen:
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        if not os.path.exists(code_gen_path):
            raise ValueError(f"{model_id} does not exist")
        return cls.from_yaml(code_gen_path)

    def validate(self) -> Optional[str]:
        if self.is_aimet and self.use_hub_quantization:
            return "Flags is_aimet and use_hub_quantization cannot both be set."
        if self.use_hub_quantization and not self.eval_datasets:
            return "Must set eval_datasets if use_hub_quantization is set."
        return None

    @classmethod
    def from_yaml(cls: type[QAIHMModelCodeGen], path: str | Path) -> QAIHMModelCodeGen:
        if not path or not os.path.exists(path):
            return QAIHMModelCodeGen()  # Default Schema
        return super().from_yaml(path)


@dataclass
class QAIHMModelInfo(BaseDataClass):
    # Name of the model as it will appear on the website.
    # Should have dashes instead of underscores and all
    # words capitalized. For example, `Whisper-Base-En`.
    name: str

    # Name of the model's folder within the repo.
    id: str

    # Whether or not the model is published on the website.
    # This should be set to public unless the model has poor accuracy/perf.
    status: MODEL_STATUS

    # Device form factors for which we don't publish performance data.
    private_perf_form_factors: list[ScorecardDevice.FormFactor]

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN

    # A 2-3 sentence description of how the model can be used.
    description: str

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE

    # A list of applicable tags to add to the model
    tags: list[MODEL_TAG]

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: list[str]

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: list[str]

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: list[ScorecardDevice.FormFactor]

    # Whether the model has a static image uploaded in S3. All public models must have this.
    has_static_banner: bool

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool

    # CodeGen options from code-gen.yaml in the model's folder.
    code_gen_config: QAIHMModelCodeGen

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: list[str]

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #       This and `Number of parameters` should be auto-generated by running `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_name>`
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: dict[str, str]

    # The license type of the original model repo.
    license_type: str

    # Some models are made by company
    model_maker_id: Optional[str] = None

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: Optional[str] = None

    # The title of the research paper.
    research_paper_title: Optional[str] = None

    # A link to the original github repo with the model's code.
    source_repo: Optional[str] = None

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: Optional[str] = None

    # A link to the AIHub license, unless the license is more restrictive like GPL.
    # In that case, this should point to the same as the model license.
    deploy_license: Optional[str] = None

    # Should be set to `AI Model Hub License`, unless the license is more restrictive like GPL.
    # In that case, this should be the same as the model license.
    deploy_license_type: Optional[str] = None

    # If set, model assets shouldn't distributed.
    restrict_model_sharing: bool = False

    # If status is private, this must have a reference to an internal issue with an explanation.
    status_reason: Optional[str] = None

    # If the model outputs class indices, this field should be set and point
    # to a file in `qai_hub_models/labels`, which specifies the name for each index.
    labels_file: Optional[str] = None

    # It is a large language model (LLM) or not.
    model_type_llm: bool = False

    @dataclass
    class LLMDetails(BaseDataClass):
        @unique
        class CALL_TO_ACTION(Enum):
            DOWNLOAD = 0
            VIEW_README = 1
            CONTACT_FOR_PURCHASE = 2
            CONTACT_FOR_DOWNLOAD = 3

            @staticmethod
            def from_string(string: str) -> QAIHMModelInfo.LLMDetails.CALL_TO_ACTION:
                return QAIHMModelInfo.LLMDetails.CALL_TO_ACTION[
                    string.upper().replace(" ", "_")
                ]

            def __str__(self):
                return self.name.title().replace("_", " ")

        @dataclass
        class LLMDetailDeviceRuntime(BaseDataClass):
            model_download_url: str

            # If undefined, the global "genie_compatible" parameter in LLMDetails must not be None.
            genie_compatible: Optional[bool] = None

        call_to_action: QAIHMModelInfo.LLMDetails.CALL_TO_ACTION

        # If undefined, this must be set per-device in the "devices" parameter below.
        genie_compatible: Optional[bool] = None

        # Dict<Device Name, Dict<Long Runtime Name, LLMDetailDeviceRuntime>
        devices: Optional[
            dict[
                str,
                dict[
                    ScorecardProfilePath,
                    QAIHMModelInfo.LLMDetails.LLMDetailDeviceRuntime,
                ],
            ]
        ] = None

        @classmethod
        def from_dict(
            cls: type[QAIHMModelInfo.LLMDetails], val_dict: dict[str, Any]
        ) -> QAIHMModelInfo.LLMDetails:
            sanitized_dict: dict[str, Any] = {}
            sanitized_dict[
                "call_to_action"
            ] = QAIHMModelInfo.LLMDetails.CALL_TO_ACTION.from_string(
                val_dict["call_to_action"]
            )
            if genie_compatible := val_dict.get("genie_compatible"):
                sanitized_dict["genie_compatible"] = genie_compatible

            for key in val_dict:
                if key in sanitized_dict:
                    continue

                device_name = key
                runtime_config_mapping: dict[str, dict[str, str]] = val_dict[
                    device_name
                ]
                assert isinstance(runtime_config_mapping, dict)
                processed_per_runtime_config_mapping = {}
                for runtime_name, runtime_config in runtime_config_mapping.items():
                    assert isinstance(runtime_config, dict)
                    runtime = None
                    for path in ScorecardProfilePath:
                        if path.long_name == runtime_name or path.name == runtime_name:
                            runtime = path
                            break

                    if not runtime:
                        raise ValueError(
                            f"Unknown runtime specified in LLM details for device {device_name}: {runtime}"
                        )

                    processed_per_runtime_config_mapping[
                        runtime
                    ] = QAIHMModelInfo.LLMDetails.LLMDetailDeviceRuntime.from_dict(
                        runtime_config
                    )

                devices_dict = sanitized_dict.get("devices", None)
                if not devices_dict:
                    devices_dict = {}
                    sanitized_dict["devices"] = devices_dict

                devices_dict[device_name] = processed_per_runtime_config_mapping

            return super().from_dict(sanitized_dict)

        def validate(self) -> Optional[str]:
            if self.genie_compatible is None:
                if self.devices:
                    for device_runtime_config_mapping in self.devices.values():
                        for runtime_detail in device_runtime_config_mapping.values():
                            if (
                                self.call_to_action
                                == QAIHMModelInfo.LLMDetails.CALL_TO_ACTION.CONTACT_FOR_PURCHASE
                                and runtime_detail.genie_compatible
                            ):
                                return "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."
                            elif runtime_detail.genie_compatible is None:
                                return "In LLM details, if genie_compatible is None, it must be set for each runtime on each provided device."
            else:
                if (
                    self.call_to_action
                    == QAIHMModelInfo.LLMDetails.CALL_TO_ACTION.CONTACT_FOR_PURCHASE
                    and self.genie_compatible
                ):
                    return "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."

                if self.devices:
                    for device_runtime_config_mapping in self.devices.values():
                        for runtime_detail in device_runtime_config_mapping.values():
                            if runtime_detail.genie_compatible is not None:
                                return "In LLM details, if genie_compatible is not None, it cannot be set separately per-device."

            return None

    # Add per device, download, app and if the model is available for purchase.
    llm_details: Optional[QAIHMModelInfo.LLMDetails] = None

    @property
    def is_quantized(self) -> bool:
        return (
            self.code_gen_config.is_aimet or self.code_gen_config.use_hub_quantization
        )

    def validate(self) -> Optional[str]:
        """Returns false with a reason if the info spec for this model is not valid."""
        # Validate ID
        if self.id not in MODEL_IDS:
            return f"{self.id} is not a valid QAI Hub Models ID."
        if " " in self.id or "-" in self.id:
            return "Model IDs cannot contain spaces or dashes."
        if self.id.lower() != self.id:
            return "Model IDs must be lowercase."

        # Validate (used as repo name for HF as well)
        if " " in self.name:
            return "Model Name must not have a space."

        # Headline should end with period
        if not self.headline.endswith("."):
            return "Model headlines must end with a period."

        # Quantized models must contain quantized tag
        if ("quantized" in self.id) and (MODEL_TAG.QUANTIZED not in self.tags):
            return f"Quantized models must have quantized tag. tags: {self.tags}"
        if ("quantized" not in self.id) and (MODEL_TAG.QUANTIZED in self.tags):
            return f"Models with a quantized tag must have 'quantized' in the id. tags: {self.tags}"

        # Validate related models are present
        for r_model in self.related_models:
            if r_model not in MODEL_IDS:
                return f"Related model {r_model} is not a valid model ID."
            if r_model == self.id:
                return f"Model {r_model} cannot be related to itself."

        # If paper is arxiv, it should be an abs link
        if self.research_paper is not None and self.research_paper.startswith(
            "https://arxiv.org/"
        ):
            if "/abs/" not in self.research_paper:
                return "Arxiv links should be `abs` links, not link directly to pdfs."

        # If license_type does not match the map, return an error
        if self.license_type not in HF_AVAILABLE_LICENSES:
            return f"license can be one of these: {HF_AVAILABLE_LICENSES}"

        # Status Reason
        if self.status == MODEL_STATUS.PRIVATE and not self.status_reason:
            return "Private models must set `status_reason` in info.yaml with a link to the related issue."

        if self.status == MODEL_STATUS.PUBLIC and self.status_reason:
            return "`status_reason` in info.yaml should not be set for public models."

        # Labels file
        if self.labels_file is not None:
            if not os.path.exists(ASSET_CONFIG.get_labels_file_path(self.labels_file)):
                return f"Invalid labels file: {self.labels_file}"

        # Required assets exist
        if self.status == MODEL_STATUS.PUBLIC:
            if not os.path.exists(self.get_package_path() / "info.yaml"):
                return "All public models must have an info.yaml"

            if (
                self.code_gen_config.tflite_export_failure_reason
                and self.code_gen_config.qnn_export_failure_reason
                and self.code_gen_config.onnx_export_failure_reason
            ):
                return "Public models must support at least one export path"

            if not self.has_static_banner:
                return "Public models must have a static asset."

        session = create_session()
        if self.has_static_banner:
            static_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.STATIC_IMG
            )
            if session.head(static_banner_url).status_code != requests.codes.ok:
                return f"Static banner is missing at {static_banner_url}"
        if self.has_animated_banner:
            animated_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.ANIMATED_MOV
            )
            if session.head(animated_banner_url).status_code != requests.codes.ok:
                return f"Animated banner is missing at {animated_banner_url}"

        expected_qaihm_repo = Path("qai_hub_models") / "models" / self.id
        if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(self.id):
            return "QAIHM repo not pointing to expected relative path"

        expected_example_use = f"qai_hub_models/models/{self.id}#example--usage"
        if expected_example_use != ASSET_CONFIG.get_example_use(self.id):
            return "Example-usage field not pointing to expected relative path"

        purchase_required = False

        # Check that model_type_llm and llm_details fields
        if self.model_type_llm:
            if not self.llm_details:
                return "llm_details must be set if model type is LLM"

            if bad_llm_details := self.llm_details.validate():
                return bad_llm_details

            purchase_required = (
                self.llm_details.call_to_action
                == QAIHMModelInfo.LLMDetails.CALL_TO_ACTION.CONTACT_FOR_PURCHASE
            )

            # Download URL can only be validated in a scope with a model ID, so this
            # is validated here rather than on the LLM details class' validator.
            if self.llm_details.devices:
                for device_runtime_config_mapping in self.llm_details.devices.values():
                    for runtime_detail in device_runtime_config_mapping.values():
                        version = runtime_detail.model_download_url.split("/")[0][1:]
                        relative_path = "/".join(
                            runtime_detail.model_download_url.split("/")[1:]
                        )
                        model_download_url = ASSET_CONFIG.get_model_asset_url(
                            self.id, version, relative_path
                        )
                        if (
                            session.head(model_download_url).status_code
                            != requests.codes.ok
                        ):
                            return f"Download URL is missing at {runtime_detail.model_download_url}"
        elif self.llm_details:
            return "Model type must be LLM if llm_details is set"

        if not self.deploy_license and not purchase_required:
            return "deploy_license cannot be empty"

        if not self.deploy_license_type and not purchase_required:
            return "deploy_license_type cannot be empty"

        # Check code gen
        return self.code_gen_config.validate()

    def get_package_name(self):
        return f"{QAIHM_PACKAGE_NAME}.{MODELS_PACKAGE_NAME}.{self.id}"

    def get_package_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return get_qaihm_models_root(root) / self.id

    def get_model_definition_path(self):
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "model.py"
        )

    def get_demo_path(self):
        return os.path.join(
            ASSET_CONFIG.get_qaihm_repo(self.id, relative=False), "demo.py"
        )

    def get_labels_file_path(self):
        if self.labels_file is None:
            return None
        return ASSET_CONFIG.get_labels_file_path(self.labels_file)

    def get_info_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "info.yaml"

    def get_hf_pipeline_tag(self):
        return self.use_case.map_to_hf_pipeline_tag()

    def get_hugging_face_metadata(self, root: Path = QAIHM_PACKAGE_ROOT):
        # Get the metadata for huggingface model cards.
        hf_metadata: dict[str, Union[str, list[str]]] = dict()
        hf_metadata["library_name"] = "pytorch"
        hf_metadata["license"] = self.license_type
        hf_metadata["tags"] = [tag.name.lower() for tag in self.tags] + ["android"]
        hf_metadata["pipeline_tag"] = self.get_hf_pipeline_tag()
        return hf_metadata

    def get_model_details(self):
        # Model details.
        details = (
            "- **Model Type:** "
            + self.use_case.__str__().lower().capitalize()
            + "\n- **Model Stats:**"
        )
        for name, val in self.technical_details.items():
            details += f"\n  - {name}: {val}"
        return details

    def get_perf_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "perf.yaml"

    def get_code_gen_yaml_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "code-gen.yaml"

    def get_readme_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "README.md"

    def get_hf_model_card_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "HF_MODEL_CARD.md"

    def get_requirements_path(self, root: Path = QAIHM_PACKAGE_ROOT):
        return self.get_package_path(root) / "requirements.txt"

    def has_model_requirements(self, root: Path = QAIHM_PACKAGE_ROOT):
        return os.path.exists(self.get_requirements_path(root))

    @classmethod
    def from_model(cls: type[QAIHMModelInfo], model_id: str) -> QAIHMModelInfo:
        schema_path = QAIHM_MODELS_ROOT / model_id / "info.yaml"
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        if not os.path.exists(schema_path):
            raise ValueError(f"{model_id} does not exist")
        return cls.from_yaml_and_code_gen(schema_path, code_gen_path)

    @classmethod
    def from_dict(
        cls: type[QAIHMModelInfo], val_dict: dict[str, Any]
    ) -> QAIHMModelInfo:
        val_dict["status"] = MODEL_STATUS.from_string(val_dict["status"])
        val_dict["private_perf_form_factors"] = [
            ScorecardDevice.FormFactor.from_string(tag)
            for tag in val_dict.get("private_perf_form_factors", [])
        ]
        val_dict["domain"] = MODEL_DOMAIN.from_string(val_dict["domain"])
        val_dict["use_case"] = MODEL_USE_CASE.from_string(val_dict["use_case"])
        val_dict["tags"] = [MODEL_TAG.from_string(tag) for tag in val_dict["tags"]]
        val_dict["form_factors"] = [
            ScorecardDevice.FormFactor.from_string(tag)
            for tag in val_dict["form_factors"]
        ]
        if llm_details := val_dict.get("llm_details", None):
            val_dict["llm_details"] = QAIHMModelInfo.LLMDetails.from_dict(llm_details)

        code_gen_config = val_dict.get("code_gen_config", None)
        if isinstance(code_gen_config, dict):
            val_dict["code_gen_config"] = QAIHMModelCodeGen.from_dict(code_gen_config)
        elif not isinstance(code_gen_config, QAIHMModelCodeGen):
            raise ValueError("code_gen_config must be dict or QAIHMModelCodeGen")

        return super().from_dict(val_dict)

    @classmethod
    def from_yaml_and_code_gen(
        cls: type[QAIHMModelInfo], info_path: str | Path, code_gen_path: str | Path
    ) -> QAIHMModelInfo:
        info = load_yaml(info_path)
        info["code_gen_config"] = QAIHMModelCodeGen.from_yaml(code_gen_path)
        return cls.from_dict(info)
