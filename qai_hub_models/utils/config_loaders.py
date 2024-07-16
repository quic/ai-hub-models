# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

import requests
from qai_hub.util.session import create_session
from schema import And
from schema import Optional as OptionalSchema
from schema import Schema, SchemaError

from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET, load_yaml
from qai_hub_models.utils.base_model import TargetRuntime
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


class FORM_FACTOR(Enum):
    PHONE = 0
    TABLET = 1
    IOT = 2
    XR = 3

    @staticmethod
    def from_string(string: str) -> "FORM_FACTOR":
        return FORM_FACTOR[string.upper()]

    def __str__(self):
        if self == FORM_FACTOR.IOT:
            return "IoT"
        return self.name.title()


class MODEL_DOMAIN(Enum):
    COMPUTER_VISION = 0
    AUDIO = 1
    MULTIMODAL = 2
    GENERATIVE_AI = 3

    @staticmethod
    def from_string(string: str) -> "MODEL_DOMAIN":
        return MODEL_DOMAIN[string.upper().replace(" ", "_")]

    def __str__(self):
        return self.name.title().replace("_", " ")


class MODEL_TAG(Enum):
    BACKBONE = 0
    REAL_TIME = 1
    FOUNDATION = 2
    QUANTIZED = 3
    LLM = 4
    GENERATIVE_AI = 5

    @staticmethod
    def from_string(string: str) -> "MODEL_TAG":
        assert "_" not in string
        return MODEL_TAG[string.upper().replace("-", "_")]

    def __str__(self) -> str:
        return self.name.replace("_", "-").lower()

    def __repr__(self) -> str:
        return self.__str__()


def is_gen_ai_model(tags: List[MODEL_TAG]) -> bool:
    return MODEL_TAG.LLM in tags or MODEL_TAG.GENERATIVE_AI in tags


class MODEL_STATUS(Enum):
    PUBLIC = 0
    PRIVATE = 1
    # proprietary models are released only internally
    PROPRIETARY = 2

    @staticmethod
    def from_string(string: str) -> "MODEL_STATUS":
        return MODEL_STATUS[string.upper()]

    def __str__(self):
        return self.name


class MODEL_USE_CASE(Enum):
    # Image: 100 - 199
    IMAGE_CLASSIFICATION = 100
    IMAGE_EDITING = 101
    IMAGE_GENERATION = 102
    SUPER_RESOLUTION = 103
    SEMANTIC_SEGMENTATION = 104
    DEPTH_ESTIMATION = 105
    # Ex: OCR, image caption
    IMAGE_TO_TEXT = 105
    OBJECT_DETECTION = 106
    POSE_ESTIMATION = 107

    # Audio: 200 - 299
    SPEECH_RECOGNITION = 200
    AUDIO_ENHANCEMENT = 201

    # Video: 300 - 399
    VIDEO_CLASSIFICATION = 300
    VIDEO_GENERATION = 301

    # LLM: 400 - 499
    TEXT_GENERATION = 400

    @staticmethod
    def from_string(string: str) -> "MODEL_USE_CASE":
        return MODEL_USE_CASE[string.upper().replace(" ", "_")]

    def __str__(self):
        return self.name.replace("_", " ").title()

    def map_to_hf_pipeline_tag(self):
        """Map our usecase to pipeline-tag used by huggingface."""
        if self.name in {"IMAGE_EDITING", "SUPER_RESOLUTION"}:
            return "image-to-image"
        if self.name == "SEMANTIC_SEGMENTATION":
            return "image-segmentation"
        if self.name == "POSE_ESTIMATION":
            return "image-classification"
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


@dataclass
class ModelRuntimePerformanceDetails:
    model_name: str
    device_name: str
    device_os: str
    runtime: TargetRuntime
    inference_time_ms: int
    peak_memory_bytes: Tuple[int, int]  # min, max
    compute_unit_counts: Dict[str, int]


class QAIHMModelPerf:
    """Class to read the perf.yaml and parse it for displaying it on HuggingFace."""

    def __init__(self, perf_yaml_path, model_name):
        self.model_name = model_name
        self.perf_yaml_path = perf_yaml_path
        self.skip_overall = False
        self.skip_tflite = False
        self.skip_qnn = False
        self.tflite_row = (
            "| Samsung Galaxy S23 Ultra (Android 13) | Snapdragon® 8 Gen 2 |"
        )
        self.qnn_row = "| Samsung Galaxy S23 Ultra (Android 13) | Snapdragon® 8 Gen 2 |"

        if os.path.exists(self.perf_yaml_path):
            self.perf_details = load_yaml(self.perf_yaml_path)
            num_models = len(self.perf_details["models"])

            # Get TFLite summary from perf.yaml
            try:
                self.tflite_summary = []
                for model in self.perf_details["models"]:
                    self.tflite_summary.append(
                        model["performance_metrics"][0][TFLITE_PATH]
                    )
            except Exception:
                self.skip_tflite = True

            if not self.skip_overall and not self.skip_tflite:
                for num in range(num_models):
                    if isinstance(self.tflite_summary[num]["inference_time"], str):
                        self.skip_tflite = True

            # Get QNN summary from perf.yaml
            try:
                self.qnn_summary = []
                for model in self.perf_details["models"]:
                    self.qnn_summary.append(model["performance_metrics"][0][QNN_PATH])
            except Exception:
                self.skip_qnn = True
            if not self.skip_overall and not self.skip_qnn:
                for num in range(num_models):
                    if isinstance(self.qnn_summary[num]["inference_time"], str):
                        self.skip_qnn = True
        else:
            self.skip_overall = True

    def _get_runtime_type(self, model_type):
        if model_type == "tflite":
            return "TFLite"
        if model_type == "so":
            return "QNN Model Library"
        if model_type == "bin":
            return "QNN Binary"
        raise RuntimeError(f"Unsupported model_type specified {model_type}.")

    def get_row(self, skip, summary_list, initial_row, model_type, has_assets=True):
        # Creating a row for performance table.
        row = ""
        if not skip:
            names = self.get_submodel_names()
            for summary, name in zip(summary_list, names):
                inf_time = summary["inference_time"]
                inference_time = f"{inf_time / 1000} ms"
                mem_min = bytes_to_mb(summary["estimated_peak_memory_range"]["min"])
                mem_max = bytes_to_mb(summary["estimated_peak_memory_range"]["max"])
                peak_memory_range = f"{mem_min} - {mem_max} MB"
                if model_type == "tflite":
                    self.tflite_inference_time = inference_time
                    self.tflite_peak_memory_range = peak_memory_range
                elif model_type == "so" or model_type == "bin":
                    self.qnn_inference_time = inference_time
                    self.qnn_peak_memory_range = peak_memory_range
                primary_compute_unit = summary["primary_compute_unit"]
                precision = summary["precision"].upper()
                base_url = ASSET_CONFIG.get_hugging_face_url(self.model_name)
                # For no_assets models, only show model name and no-link
                # as there is not target model to download
                if has_assets:
                    target_model = f" [{name}.{model_type}]({base_url}/blob/main/{name}.{model_type})"
                else:
                    target_model = name

                runtime_type = self._get_runtime_type(model_type)
                row += (
                    initial_row
                    + f" {runtime_type} | {inference_time} | {peak_memory_range} | {precision} | {primary_compute_unit} | {target_model} \n"
                )
            return row
        return ""

    def get_tflite_row(self):
        # Get TFLite row for a submodel on a device.
        return self.get_row(
            self.skip_tflite, self.tflite_summary, self.tflite_row, "tflite"
        )

    def get_qnn_row(self, is_precompiled: bool = False, has_assets=True):
        # Get QNN row for a submodel on a device.
        return self.get_row(
            self.skip_qnn,
            self.qnn_summary,
            self.qnn_row,
            "bin" if is_precompiled else "so",
            has_assets,
        )

    def body_perf(self, is_precompiled: bool = False, has_assets: bool = True):
        # Combine all the rows to make the body of performance table.
        if self.skip_tflite:
            return self.get_qnn_row(is_precompiled, has_assets)
        elif self.skip_qnn:
            return self.get_tflite_row()
        else:
            return self.get_tflite_row() + self.get_qnn_row(is_precompiled, has_assets)

    def compute_unit_summary(self, runtime_path=TFLITE_PATH):
        # Get compute unit summary for export script's output.
        npu, gpu, cpu = 0, 0, 0
        cu_summary = ""
        for model in self.perf_details["models"]:
            layer_info = model["performance_metrics"][0][runtime_path]["layer_info"]
            npu += layer_info["layers_on_npu"]
            gpu += layer_info["layers_on_gpu"]
            cpu += layer_info["layers_on_cpu"]
        if npu > 0:
            cu_summary += f"NPU ({npu})"
        if gpu > 0:
            cu_summary += f"GPU ({gpu})"
        if cpu > 0:
            cu_summary += f"CPU ({cpu})"
        return cu_summary

    def get_submodel_names_and_ids(self):
        # Get the names, TFLite job ids and QNN job ids.
        names = self.get_submodel_names()
        tflite_job_ids, qnn_job_ids = [], []
        for model in self.perf_details["models"]:
            if TFLITE_PATH in model["performance_metrics"][0]:
                tflite_job_ids.append(
                    model["performance_metrics"][0][TFLITE_PATH]["job_id"]
                )
            if QNN_PATH in model["performance_metrics"][0]:
                qnn_job_ids.append(model["performance_metrics"][0][QNN_PATH]["job_id"])
        return names, tflite_job_ids, qnn_job_ids

    def get_submodel_names(self):
        # Get names of all the submodels.
        names = []
        for model in self.perf_details["models"]:
            names.append(model["name"])
        return names

    def get_perf_details(
        self,
        runtime: TargetRuntime,
        device: str | None = None,
        device_os: str | None = None,
    ) -> Dict[str, ModelRuntimePerformanceDetails | None]:
        """
        Get model performance details for the selected device and runtime.

        If device is None, picks the first device specified in the perf results.

        Returns a dictionary of
            { model_component_name : performance details object }

        If there is only one component, model_component_name == model_name.

        The performance details object will be null if the requested
        perf details do not exist, or if the perf job failed.
        """
        if runtime == TargetRuntime.TFLITE:
            rt_name = "torchscript_onnx_tflite"
        elif runtime == TargetRuntime.QNN:
            rt_name = "torchscript_onnx_qnn"
        else:
            raise NotImplementedError()

        # Model -> Performance Details
        # None == Test did not run.
        perf_details: Dict[str, ModelRuntimePerformanceDetails | None] = {}

        for model in self.perf_details["models"]:
            name = model["name"]
            metrics = model["performance_metrics"]
            for device_metrics in metrics:
                device_name = device_metrics["reference_device_info"]["name"]
                metric_device_os = device_metrics["reference_device_info"]["os"]

                # Verify Device Matches Requested Device
                if device and device_name != device:
                    continue
                if device_os and metric_device_os != device_os:
                    continue

                perf_rt = device_metrics.get(rt_name, None)

                # Inference Time
                inf_time = perf_rt["inference_time"] if perf_rt else "null"
                if inf_time == "null":
                    # Compilation or inference failed.
                    perf_details[name] = None
                    continue
                inf_time /= 1000

                # Memory
                peak_mem = perf_rt["estimated_peak_memory_range"]
                peak_mem_bytes: Tuple[int, int] = tuple([peak_mem["min"], peak_mem["max"]])  # type: ignore

                # Layer Info
                layer_info = perf_rt["layer_info"]
                compute_unit_counts = {}
                for layer_name, count in layer_info.items():
                    if "layers_on" in layer_name:
                        if count > 0:
                            compute_unit_counts[layer_name[-3:].upper()] = count

                perf_details[name] = ModelRuntimePerformanceDetails(
                    model_name=model,
                    device_name=device_name,
                    device_os=metric_device_os,
                    runtime=runtime,
                    inference_time_ms=inf_time,
                    peak_memory_bytes=peak_mem_bytes,
                    compute_unit_counts=compute_unit_counts,
                )

            if name not in perf_details.keys():
                perf_details[name] = None

        return perf_details


def _get_origin(input_type: Type) -> Type:
    """
    For nested types like List[str] or Union[str, int], this function will
        return the "parent" type like List or Union.

    If the input type is not a nested type, the function returns the input_type.
    """
    return getattr(input_type, "__origin__", input_type)


def _extract_optional_type(input_type: Type) -> Type:
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


def _constructor_from_type(input_type: Type) -> Union[Type, Callable]:
    """
    Given a type, return the appropriate constructor for that type.

    For primitive types like str and int, the type and constructor are the same object.

    For types like List, the constructor is list.
    """
    input_type = _get_origin(input_type)
    if input_type == List:
        return list
    if input_type == Dict:
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
            else:
                field_key = field.name
            schema_dict[field_key] = _constructor_from_type(field_type)
        return Schema(And(schema_dict))

    @classmethod
    def from_dict(
        cls: Type[BaseDataClassTypeVar], val_dict: Dict[str, Any]
    ) -> BaseDataClassTypeVar:
        kwargs = {field.name: val_dict[field.name] for field in fields(cls)}
        return cls(**kwargs)


BaseDataClassTypeVar = TypeVar("BaseDataClassTypeVar", bound="BaseDataClass")


@dataclass
class QAIHMModelCodeGen(BaseDataClass):
    # Whether the model is quantized with aimet.
    is_aimet: bool = False

    # Whether the model's demo supports running on device with the `--on-device` flag.
    has_on_target_demo: bool = False

    # If the model doesn't work on qnn, this should explain why,
    # ideally with a reference to an internal issue.
    qnn_export_failure_reason: str = ""

    # If the model doesn't work on tflite, this should explain why,
    # ideally with a reference to an internal issue.
    tflite_export_failure_reason: str = ""

    # If the model doesn't work on onnx, this should explain why,
    # ideally with a reference to an internal issue.
    onnx_export_failure_reason: str = ""

    # Sets the `check_trace` argument on `torch.jit.trace`.
    check_trace: bool = True

    # A list of input names to transpose to channel last format for perf reasons.
    channel_last_input: Optional[List[str]] = None

    # A list of output names to transpose to channel last format for perf reasons.
    channel_last_output: Optional[List[str]] = None

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
    outputs_to_skip_validation: Optional[List[str]] = None

    # Additional arguments to initialize the model when unit testing export.
    # This is commonly used to test a smaller variant in the unit test.
    export_test_model_kwargs: Optional[Dict[str, str]] = None

    # Some models are comprised of submodels that should be compiled separately.
    # For example, this is used when there is an encoder/decoder pattern.
    # This is a dict from component name to a python expression that can be evaluated
    # to produce the submodel. The expression can assume the parent model has been
    # initialized and assigned to the variable `model`.
    components: Optional[Dict[str, Any]] = None

    # If components is set, this field can specify a subset of components to run
    # by default when invoking `export.py`. If unset, all components are run by default.
    default_components: Optional[List[str]] = None

    # If set, skips generating `test_generated.py`, which is used for unit-testing
    # `export.py` as well as running the weekly scorecard.
    skip_tests: bool = False

    # Whether the model uses the pre-compiled pattern instead of the
    # standard pre-trained pattern.
    is_precompiled: bool = False

    # If set, disables uploading compiled assets to HuggingFace
    no_assets: bool = False

    # If set, disables generating `export.py`.
    skip_export: bool = False

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    # A list of optimizations from `torch.utils.mobile_optimizer` that will
    # speed up the conversion to torchscript.
    torchscript_opt: Optional[List[str]] = None

    # A comma separated list of metrics to print in the inference summary of `export.py`.
    inference_metrics: str = "psnr"

    # Additional details that can be set on the model's readme.
    additional_readme_section: str = ""

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # If set, generates an `evaluate.py` file which can be used to evaluate the model
    # on a full dataset. Datasets specified here must be chosen from `qai_hub_models/datasets`.
    eval_datasets: Optional[List[str]] = None

    @classmethod
    def from_model(cls: Type[QAIHMModelCodeGen], model_id: str) -> QAIHMModelCodeGen:
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        if not os.path.exists(code_gen_path):
            raise ValueError(f"{model_id} does not exist")
        return cls.from_yaml(code_gen_path)

    @classmethod
    def from_yaml(
        cls: Type[QAIHMModelCodeGen], code_gen_path: str | Path | None = None
    ) -> QAIHMModelCodeGen:
        # Load CFG and params
        code_gen_config = QAIHMModelCodeGen.load_code_gen_yaml(code_gen_path)
        return cls.from_dict(code_gen_config)

    @staticmethod
    def load_code_gen_yaml(path: str | Path | None = None) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return QAIHMModelCodeGen.get_schema().validate({})  # Default Schema
        data = load_yaml(path)
        try:
            # Validate high level-schema
            data = QAIHMModelCodeGen.get_schema().validate(data)
        except SchemaError as e:
            assert 0, f"{e.code} in {path}"
        return data


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

    # A brief catchy headline explaining what the model does and why it may be interesting
    headline: str

    # The domain the model is used in such as computer vision, audio, etc.
    domain: MODEL_DOMAIN

    # A 2-3 sentence description of how the model can be used.
    description: str

    # What task the model is used to solve, such as object detection, classification, etc.
    use_case: MODEL_USE_CASE

    # A list of applicable tags to add to the model
    tags: List[MODEL_TAG]

    # Link to the research paper where the model was first published. Usually an arxiv link.
    research_paper: str

    # The title of the research paper.
    research_paper_title: str

    # A link to the model's license. Most commonly found in the github repo it was cloned from.
    license: str

    # A link to the AIHub license, unless the license is more restrictive like GPL.
    # In that case, this should point to the same as the model license.
    deploy_license: str

    # A link to the original github repo with the model's code.
    source_repo: str

    # A list of real-world applicaitons for which this model could be used.
    # This is free-from and almost anything reasonable here is fine.
    applicable_scenarios: List[str]

    # A list of other similar models in the repo.
    # Typically, any model that performs the same task is fine.
    # If nothing fits, this can be left blank. Limit to 3 models.
    related_models: List[str]

    # A list of device types for which this model could be useful.
    # If unsure what to put here, default to `Phone` and `Tablet`.
    form_factors: List[FORM_FACTOR]

    # Whether the model has a static image uploaded in S3. All public models must have this.
    has_static_banner: bool

    # Whether the model has an animated asset uploaded in S3. This is optional.
    has_animated_banner: bool

    # CodeGen options from code-gen.yaml in the model's folder.
    code_gen_config: QAIHMModelCodeGen

    # The license type of the original model repo.
    license_type: str

    # Should be set to `AI Model Hub License`, unless the license is more restrictive like GPL.
    # In that case, this should be the same as the model license.
    deploy_license_type: str

    # A list of datasets for which the model has pre-trained checkpoints
    # available as options in `model.py`. Typically only has one entry.
    dataset: List[str]

    # A list of a few technical details about the model.
    #   Model checkpoint: The name of the downloaded model checkpoint file.
    #   Input resolution: The size of the model's input. For example, `2048x1024`.
    #   Number of parameters: The number of parameters in the model.
    #   Model size: The file size of the downloaded model asset.
    #       This and `Number of parameters` should be auto-generated by running `python qai_hub_models/scripts/autofill_info_yaml.py -m <model_name>`
    #   Number of output classes: The number of classes the model can classify or annotate.
    technical_details: Dict[str, str]

    # If status is private, this must have a reference to an internal issue with an explanation.
    status_reason: Optional[str] = None

    # If the model outputs class indices, this field should be set and point
    # to a file in `qai_hub_models/labels`, which specifies the name for each index.
    labels_file: Optional[str] = None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Returns false with a reason if the info spec for this model is not valid."""
        # Validate ID
        if self.id not in MODEL_IDS:
            return False, f"{self.id} is not a valid QAI Hub Models ID."
        if " " in self.id or "-" in self.id:
            return False, "Model IDs cannot contain spaces or dashes."
        if self.id.lower() != self.id:
            return False, "Model IDs must be lowercase."

        # Validate (used as repo name for HF as well)
        if " " in self.name:
            return False, "Model Name must not have a space."

        # Headline should end with period
        if not self.headline.endswith("."):
            return False, "Model headlines must end with a period."

        # Quantized models must contain quantized tag
        if ("quantized" in self.id) and (MODEL_TAG.QUANTIZED not in self.tags):
            return False, f"Quantized models must have quantized tag. tags: {self.tags}"
        if ("quantized" not in self.id) and (MODEL_TAG.QUANTIZED in self.tags):
            return (
                False,
                f"Models with a quantized tag must have 'quantized' in the id. tags: {self.tags}",
            )

        # Validate related models are present
        for r_model in self.related_models:
            if r_model not in MODEL_IDS:
                return False, f"Related model {r_model} is not a valid model ID."
            if r_model == self.id:
                return False, f"Model {r_model} cannot be related to itself."

        # If paper is arxiv, it should be an abs link
        if self.research_paper.startswith("https://arxiv.org/"):
            if "/abs/" not in self.research_paper:
                return (
                    False,
                    "Arxiv links should be `abs` links, not link directly to pdfs.",
                )

        # If license_type does not match the map, return an error
        if self.license_type not in HF_AVAILABLE_LICENSES:
            return False, f"license can be one of these: {HF_AVAILABLE_LICENSES}"

        if not self.deploy_license:
            return False, "deploy_license cannot be empty"
        if not self.deploy_license_type:
            return False, "deploy_license_type cannot be empty"

        # Status Reason
        if self.status == MODEL_STATUS.PRIVATE and not self.status_reason:
            return (
                False,
                "Private models must set `status_reason` in info.yaml with a link to the related issue.",
            )
        if self.status == MODEL_STATUS.PUBLIC and self.status_reason:
            return (
                False,
                "`status_reason` in info.yaml should not be set for public models.",
            )

        # Labels file
        if self.labels_file is not None:
            if not os.path.exists(ASSET_CONFIG.get_labels_file_path(self.labels_file)):
                return False, f"Invalid labels file: {self.labels_file}"

        # Required assets exist
        if self.status == MODEL_STATUS.PUBLIC:
            if not os.path.exists(self.get_package_path() / "info.yaml"):
                return False, "All public models must have an info.yaml"

            if (
                self.code_gen_config.tflite_export_failure_reason
                and self.code_gen_config.qnn_export_failure_reason
                and self.code_gen_config.onnx_export_failure_reason
            ):
                return False, "Public models must support at least one export path"

        session = create_session()
        if self.has_static_banner:
            static_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.STATIC_IMG
            )
            if session.head(static_banner_url).status_code != requests.codes.ok:
                return False, f"Static banner is missing at {static_banner_url}"
        if self.has_animated_banner:
            animated_banner_url = ASSET_CONFIG.get_web_asset_url(
                self.id, QAIHM_WEB_ASSET.ANIMATED_MOV
            )
            if session.head(animated_banner_url).status_code != requests.codes.ok:
                return False, f"Animated banner is missing at {animated_banner_url}"

        expected_qaihm_repo = Path("qai_hub_models") / "models" / self.id
        if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(self.id):
            return False, "QAIHM repo not pointing to expected relative path"

        expected_example_use = f"qai_hub_models/models/{self.id}#example--usage"
        if expected_example_use != ASSET_CONFIG.get_example_use(self.id):
            return False, "Example-usage field not pointing to expected relative path"

        return True, None

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
        hf_metadata: Dict[str, Union[str, List[str]]] = dict()
        hf_metadata["library_name"] = "pytorch"
        hf_metadata["license"] = self.license_type
        hf_metadata["tags"] = [tag.name.lower() for tag in self.tags] + ["android"]
        if self.dataset != []:
            hf_metadata["datasets"] = self.dataset
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
    def from_model(cls: Type[QAIHMModelInfo], model_id: str) -> QAIHMModelInfo:
        schema_path = QAIHM_MODELS_ROOT / model_id / "info.yaml"
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        if not os.path.exists(schema_path):
            raise ValueError(f"{model_id} does not exist")
        return cls.from_yaml(schema_path, code_gen_path)

    @classmethod
    def from_yaml(
        cls: Type[QAIHMModelInfo],
        info_path: str | Path,
        code_gen_path: str | Path | None = None,
    ) -> QAIHMModelInfo:
        # Load CFG and params
        data = load_yaml(info_path)
        data["status"] = MODEL_STATUS.from_string(data["status"])
        data["domain"] = MODEL_DOMAIN.from_string(data["domain"])
        data["use_case"] = MODEL_USE_CASE.from_string(data["use_case"])
        data["tags"] = [MODEL_TAG.from_string(tag) for tag in data["tags"]]
        data["form_factors"] = [
            FORM_FACTOR.from_string(tag) for tag in data["form_factors"]
        ]
        data["code_gen_config"] = QAIHMModelCodeGen.from_yaml(code_gen_path)

        try:
            # Validate high level-schema
            data = QAIHMModelInfo.get_schema().validate(data)
        except SchemaError as e:
            assert 0, f"{e.code} in {info_path}"
        return cls.from_dict(data)
