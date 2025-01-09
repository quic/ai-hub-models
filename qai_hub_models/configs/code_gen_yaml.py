# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.default_export_device import DEFAULT_EXPORT_DEVICE
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


@dataclass
class QAIHMModelCodeGen(BaseQAIHMConfig):
    """
    Schema & loader for model code-gen.yaml.
    """

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
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added.
    qnn_export_failure_reason: str = ""

    # If the model should be disabled for qnn for any reason other than
    # a job failure, this should explain why,
    # ideally with a reference to an internal issue.
    qnn_export_disable_reason: str = ""

    # If the model doesn't work on tflite, this should explain why,
    # ideally with a reference to an internal issue.
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added.
    tflite_export_failure_reason: str = ""

    # If the model should be disabled for tflite for any reason other than
    # a job failure, this should explain why,
    # ideally with a reference to an internal issue.
    tflite_export_disable_reason: str = ""

    # If the model doesn't work on onnx, this should explain why,
    # ideally with a reference to an internal issue.
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added.
    onnx_export_failure_reason: str = ""

    # If the model should be disabled for onnx for any reason other than
    # a job failure, this should explain why,
    # ideally with a reference to an internal issue.
    onnx_export_disable_reason: str = ""

    # If set, changes the default device when running export.py for the model.
    default_device: str = DEFAULT_EXPORT_DEVICE

    # Sets the `check_trace` argument on `torch.jit.trace`.
    check_trace: bool = True

    # Some model outputs have low PSNR when in practice the numerical accuracy is fine.
    # This can happen when the model outputs many low confidence values that get
    # filtered out in post-processing.
    # Omit printing PSNR in `export.py` for these to avoid confusion.
    # dict<output_idx, reason_for_skip>
    outputs_to_skip_validation: Optional[dict[int, str]] = None

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
    python_version_greater_than_or_equal_to_reason: Optional[str] = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: Optional[str] = None
    python_version_less_than_reason: Optional[str] = None

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
        if (
            self.python_version_greater_than_or_equal_to is None
            and self.python_version_greater_than_or_equal_to_reason is not None
        ):
            return "python_version_greater_than_or_equal_to_reason is set, but python_version_greater_than_or_equal_to is not."
        if (
            self.python_version_greater_than_or_equal_to is not None
            and self.python_version_greater_than_or_equal_to_reason is None
        ):
            return "python_version_greater_than_or_equal_to must have a reason (python_version_greater_than_or_equal_to_reason) set."
        if (
            self.python_version_less_than_reason is None
            and self.python_version_less_than is not None
        ):
            return "python_version_less_than must have a reason (python_version_less_than_reason) set."
        if (
            self.python_version_less_than_reason is not None
            and self.python_version_less_than is None
        ):
            return "python_version_less_than_reason is set, but python_version_less_than is not."
        return None

    @classmethod
    def from_yaml(cls: type[QAIHMModelCodeGen], path: str | Path) -> QAIHMModelCodeGen:
        if not path or not os.path.exists(path):
            return QAIHMModelCodeGen()  # Default Schema
        return super().from_yaml(path)

    def to_model_yaml(self, model_id: str):
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        self.to_yaml(code_gen_path, write_if_empty=False, delete_if_empty=True)
