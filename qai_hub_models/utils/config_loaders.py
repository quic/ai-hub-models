# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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


class QAIHMModelPerf:
    """Class to read the perf.yaml and parse it for displaying it on HuggingFace."""

    @dataclass
    class ModelRuntimePerformanceDetails:
        model_name: str
        device_name: str
        device_os: str
        runtime: TargetRuntime
        inference_time_ms: int
        peak_memory_bytes: Tuple[int, int]  # min, max
        compute_unit_counts: Dict[str, int]

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
        perf_details: Dict[
            str, QAIHMModelPerf.ModelRuntimePerformanceDetails | None
        ] = {}

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

                perf_details[name] = QAIHMModelPerf.ModelRuntimePerformanceDetails(
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


class QAIHMModelCodeGen:
    def __init__(
        self,
        is_aimet: bool,
        has_on_target_demo: bool,
        qnn_export_failure_reason: str,
        tflite_export_failure_reason: str,
        ort_export_failure_reason: str,
        check_trace: bool,
        channel_last_input: List[str],
        channel_last_output: List[str],
        outputs_to_skip_validation: List[str],
        export_test_model_kwargs: Dict[str, str],
        components: Dict[str, Any],
        default_components: List[str],
        skip_tests: bool,
        is_precompiled: bool,
        no_assets: bool,
        skip_export: bool,
        global_requirements_incompatible: bool,
        torchscript_opt: List[str],
        inference_metrics: str,
        additional_readme_section: str,
        skip_example_usage: bool,
        eval_datasets: List[str],
    ) -> None:
        self.is_aimet = is_aimet
        self.has_on_target_demo = has_on_target_demo
        self.qnn_export_failure_reason = qnn_export_failure_reason
        self.tflite_export_failure_reason = tflite_export_failure_reason
        self.ort_export_failure_reason = ort_export_failure_reason
        self.check_trace = check_trace
        self.channel_last_input = channel_last_input
        self.channel_last_output = channel_last_output
        self.outputs_to_skip_validation = outputs_to_skip_validation
        self.export_test_model_kwargs = export_test_model_kwargs
        self.components = components
        self.default_components = default_components
        self.skip_tests = skip_tests
        self.is_precompiled = is_precompiled
        self.no_assets = no_assets
        self.global_requirements_incompatible = global_requirements_incompatible
        self.torchscript_opt = torchscript_opt
        self.inference_metrics = inference_metrics
        self.additional_readme_section = additional_readme_section
        self.skip_export = skip_export
        self.skip_example_usage = skip_example_usage
        self.eval_datasets = eval_datasets

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Returns false with a reason if the info spec for this model is not valid."""
        return True, None

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
        return cls(
            code_gen_config["is_aimet"],
            code_gen_config["has_on_target_demo"],
            code_gen_config["qnn_export_failure_reason"],
            code_gen_config["tflite_export_failure_reason"],
            code_gen_config["ort_export_failure_reason"],
            code_gen_config["check_trace"],
            code_gen_config["channel_last_input"],
            code_gen_config["channel_last_output"],
            code_gen_config["outputs_to_skip_validation"],
            code_gen_config["export_test_model_kwargs"],
            code_gen_config["components"],
            code_gen_config["default_components"],
            code_gen_config["skip_tests"],
            code_gen_config["is_precompiled"],
            code_gen_config["no_assets"],
            code_gen_config["global_requirements_incompatible"],
            code_gen_config["torchscript_opt"],
            code_gen_config["inference_metrics"],
            code_gen_config["additional_readme_section"],
            code_gen_config["skip_export"],
            code_gen_config["skip_example_usage"],
            code_gen_config["eval_datasets"],
        )

    # Schema for code-gen.yaml
    CODE_GEN_YAML_SCHEMA = Schema(
        And(
            {
                OptionalSchema("has_components", default=""): str,
                OptionalSchema("is_aimet", default=False): bool,
                OptionalSchema("has_on_target_demo", default=False): bool,
                OptionalSchema("qnn_export_failure_reason", default=""): str,
                OptionalSchema("tflite_export_failure_reason", default=""): str,
                OptionalSchema("ort_export_failure_reason", default=""): str,
                OptionalSchema("check_trace", default=True): bool,
                OptionalSchema("channel_last_input", default=[]): list,
                OptionalSchema("channel_last_output", default=[]): list,
                OptionalSchema("outputs_to_skip_validation", default=[]): list,
                OptionalSchema("export_test_model_kwargs", default={}): dict,
                OptionalSchema("components", default={}): dict,
                OptionalSchema("default_components", default=[]): list,
                OptionalSchema("skip_tests", default=False): bool,
                OptionalSchema("is_precompiled", default=False): bool,
                OptionalSchema("no_assets", default=False): bool,
                OptionalSchema("global_requirements_incompatible", default=False): bool,
                OptionalSchema("torchscript_opt", default=[]): list,
                OptionalSchema("inference_metrics", default="psnr"): str,
                OptionalSchema("additional_readme_section", default=""): str,
                OptionalSchema("skip_export", default=False): bool,
                OptionalSchema("skip_example_usage", default=False): bool,
                OptionalSchema("eval_datasets", default=[]): list,
            }
        )
    )

    @staticmethod
    def load_code_gen_yaml(path: str | Path | None = None):
        if not path or not os.path.exists(path):
            return QAIHMModelCodeGen.CODE_GEN_YAML_SCHEMA.validate({})  # Default Schema
        data = load_yaml(path)
        try:
            # Validate high level-schema
            data = QAIHMModelCodeGen.CODE_GEN_YAML_SCHEMA.validate(data)
        except SchemaError as e:
            assert 0, f"{e.code} in {path}"
        return data


class QAIHMModelInfo:
    def __init__(
        self,
        name: str,
        id: str,
        status: MODEL_STATUS,
        status_reason: str | None,
        headline: str,
        domain: MODEL_DOMAIN,
        description: str,
        use_case: MODEL_USE_CASE,
        tags: List[MODEL_TAG],
        research_paper: str,
        research_paper_title: str,
        license: str,
        deploy_license: str,
        source_repo: str,
        applicable_scenarios: List[str],
        related_models: List[str],
        form_factors: List[FORM_FACTOR],
        has_static_banner: bool,
        has_animated_banner: bool,
        code_gen_config: QAIHMModelCodeGen,
        license_type: str,
        deploy_license_type: str,
        dataset: List[str],
        labels_file: str | None,
        technical_details: Dict[str, str],
    ) -> None:
        self.name = name
        self.id = id
        self.status = status
        self.status_reason = status_reason
        self.headline = headline
        self.domain = domain
        self.description = description
        self.use_case = use_case
        self.tags = tags
        self.research_paper = research_paper
        self.research_paper_title = research_paper_title
        self.license = license
        self.deploy_license = deploy_license
        self.license_type = license_type
        self.deploy_license_type = deploy_license_type
        self.dataset = dataset
        self.labels_file = labels_file
        self.source_repo = source_repo
        self.applicable_scenarios = applicable_scenarios
        self.related_models = related_models
        self.form_factors = form_factors
        self.has_static_banner = has_static_banner
        self.has_animated_banner = has_animated_banner
        self.code_gen_config = code_gen_config
        self.technical_details = technical_details

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
        info_yaml = QAIHMModelInfo.load_info_yaml(info_path)
        return cls(
            info_yaml["name"],
            info_yaml["id"],
            MODEL_STATUS.from_string(info_yaml["status"]),
            info_yaml.get("status_reason", None),
            info_yaml["headline"],
            MODEL_DOMAIN.from_string(info_yaml["domain"]),
            info_yaml["description"],
            MODEL_USE_CASE.from_string(info_yaml["use_case"]),
            [MODEL_TAG.from_string(tag) for tag in info_yaml["tags"]],
            info_yaml["research_paper"],
            info_yaml["research_paper_title"],
            info_yaml["license"],
            info_yaml["deploy_license"],
            info_yaml["source_repo"],
            info_yaml["applicable_scenarios"],
            info_yaml["related_models"],
            [FORM_FACTOR.from_string(ff) for ff in info_yaml["form_factors"]],
            info_yaml["has_static_banner"],
            info_yaml["has_animated_banner"],
            QAIHMModelCodeGen.from_yaml(code_gen_path),
            info_yaml["license_type"],
            info_yaml["deploy_license_type"],
            info_yaml["dataset"],
            info_yaml.get("labels_file", None),
            info_yaml["technical_details"],
        )

    # Schema for info.yaml
    INFO_YAML_SCHEMA = Schema(
        And(
            {
                "name": str,
                "id": str,
                "status": str,
                OptionalSchema("status_reason", default=None): str,
                "headline": str,
                "domain": str,
                "description": str,
                "use_case": str,
                "tags": lambda s: len(s) >= 0,
                "research_paper": str,
                "research_paper_title": str,
                "license": str,
                "deploy_license": str,
                "source_repo": str,
                "technical_details": dict,
                "applicable_scenarios": lambda s: len(s) >= 0,
                "related_models": lambda s: len(s) >= 0,
                "form_factors": lambda s: len(s) >= 0,
                "has_static_banner": bool,
                "has_animated_banner": bool,
                "license_type": str,
                "deploy_license_type": str,
                "dataset": list,
                OptionalSchema("labels_file", default=None): str,
            }
        )
    )

    @staticmethod
    def load_info_yaml(path: str | Path) -> Dict[str, Any]:
        data = load_yaml(path)
        try:
            # Validate high level-schema
            data = QAIHMModelInfo.INFO_YAML_SCHEMA.validate(data)
        except SchemaError as e:
            assert 0, f"{e.code} in {path}"
        return data
