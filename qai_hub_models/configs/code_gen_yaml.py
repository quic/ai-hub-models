# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qai_hub_models.models.common import Precision, TargetRuntime
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

    # The list of precisions that:
    # - Are enabled via the CLI
    # - Scorecard runs by default each week for accuracy & performance tests
    supported_precisions: list[Precision] = field(
        default_factory=lambda: [Precision.float]
    )

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
    # not be manually edited after a model is first added. If the model
    # begins to work again, this will be removed automatically by scorecard.
    qnn_scorecard_failure: str = ""

    # If the model should be disabled for qnn for any reason other than
    # a job failure, this should explain why,
    #
    # This requires a filed issue because it isn't auto-removed
    # when a model begins to work again.
    qnn_export_disable_issue: str = ""

    # If the model times out, we can't run it in testing or scorecard.
    #
    # This requires a filed issue because it isn't auto-removed
    # when a model begins to work again.
    qnn_export_timeout_issue: str = ""

    # If the model doesn't work on tflite, this should explain why,
    # ideally with a reference to an internal issue.
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added. If the model
    # begins to work again, this will be removed automatically by scorecard.
    tflite_scorecard_failure: str = ""

    # If the model should be disabled for tflite for any reason other than
    # a job failure, this should explain why.
    #
    # This requires a filed issue because it isn't auto-removed
    # when a model begins to work again.
    tflite_export_disable_issue: str = ""

    # If the model doesn't work on onnx, this should explain why,
    # ideally with a reference to an internal issue.
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added. If the model
    # begins to work again, this will be removed automatically by scorecard.
    onnx_scorecard_failure: str = ""

    # If the model should be disabled for onnx for any reason other than
    # a job failure, this should explain why.
    #
    # This requires a filed issue because it isn't auto-removed
    # when a model begins to work again.
    onnx_export_disable_issue: str = ""

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
    components: Optional[dict[str, str]] = None

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

    # Requirements that must be pre-installed before installing the general model requirements.
    #
    # Eg. for example, `pip install qai_hub_models[model]` won't work,
    # but `pip install package_a package_b ...; pip install qai_hub_models[model]` does work.
    #
    # This setting defines what "package_a package_b ..." is.
    #
    # This is required when a package needs to be built from source by pip but
    # doesn't have its requirements set up correctly.
    pip_pre_build_reqs: Optional[str] = None

    # If extra flags are needed when `pip install`ing for this model, provide them here
    pip_install_flags: Optional[str] = None

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

    # By default inference tests are done using 8gen1 chipset to avoid overloading
    # newer devices. Some models don't work on 8gen1, so use 8gen3 for those.
    inference_on_8gen3: bool = False

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: Optional[str] = None
    python_version_greater_than_or_equal_to_reason: Optional[str] = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: Optional[str] = None
    python_version_less_than_reason: Optional[str] = None

    @property
    def component_names(self) -> list[str] | None:
        if self.default_components:
            return self.default_components
        return list(self.components.keys()) if self.components else None

    def supports_runtime(self, runtime: TargetRuntime) -> bool:
        """
        Return true if this runtime is supported by this model.
        Return false if this model has a failure reason set for this runtime.
        """
        return not bool(self.runtime_failure_reason(runtime))

    def runtime_failure_reason(self, runtime: TargetRuntime) -> str | None:
        """
        Return the reason a model failed or None if the model did not fail.
        """
        if not any(runtime.supports_precision(p) for p in self.supported_precisions):
            return "Runtime does not support any precisions supported by this model"

        if runtime == TargetRuntime.PRECOMPILED_QNN_ONNX:
            runtime = (
                TargetRuntime.QNN
            )  # QNN Support is a proxy for precompiled QNN ONNX.
        automated_skip = getattr(
            self, f"{runtime.name.lower()}_scorecard_failure", None
        )
        user_provided_skip = getattr(
            self, f"{runtime.name.lower()}_export_disable_issue", None
        )
        timeout = getattr(self, f"{runtime.name.lower()}_export_timeout_issue", None)
        return user_provided_skip or timeout or automated_skip

    @classmethod
    def from_model(cls: type[QAIHMModelCodeGen], model_id: str) -> QAIHMModelCodeGen:
        model_folder = QAIHM_MODELS_ROOT / model_id
        if not os.path.exists(model_folder):
            raise ValueError(f"{model_id} does not exist")

        code_gen_path = model_folder / "code-gen.yaml"
        if not os.path.exists(code_gen_path):
            out = QAIHMModelCodeGen()
        else:
            out = cls.from_yaml(code_gen_path)

        # This is a hack so we don't have to populate all of the legacy quantized models' code gen configs
        # with the quantization type.
        #
        # TODO(#13765): remove this hack when we remove the quantized model folders
        if "_quantized" in model_id:
            if (
                len(out.supported_precisions) == 1
                and out.supported_precisions[0] == Precision.float
            ):
                if model_id.endswith("w8a16_quantized"):
                    out.supported_precisions = [Precision.w8a16]
                elif model_id.endswith("_quantized"):
                    out.supported_precisions = [Precision.w8a8]

        return out

    @property
    def has_components(self) -> bool:
        return bool(self.components)

    @property
    def can_use_quantize_job(self) -> bool:
        """
        Whether the model can be quantized via quantize job.
        This may return true even if the model does list support for non-float precisions.
        """
        return not self.is_precompiled and not self.is_aimet and not self.has_components

    @property
    def supports_quantization(self) -> bool:
        return any(x != Precision.float for x in self.supported_precisions)

    def validate(self) -> Optional[str]:
        if (
            not self.is_precompiled
            and not self.is_aimet
            and self.supports_quantization
            and not self.eval_datasets
        ):
            return "Must set eval_datasets if Hub quantization is supported"
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
        if self.pip_install_flags and not self.global_requirements_incompatible:
            return "If pip_install_flags is set, global_requirements_incompatible must also be true."
        if self.pip_pre_build_reqs and not self.global_requirements_incompatible:
            return "If pip_pre_build_reqs is set, global_requirements_incompatible must also be true."

        for runtime in TargetRuntime:
            disable_field_name = f"{runtime.name.lower()}_export_disable_issue"
            user_provided_skip = getattr(self, disable_field_name, None)

            timeout_field_name = f"{runtime.name.lower()}_export_timeout_issue"
            timeout = getattr(self, timeout_field_name, None)

            issue_link = "https://github.com/qcom-ai-hub/tetracode/issues/"
            for field_name, field_val in (
                (disable_field_name, user_provided_skip),
                (timeout_field_name, timeout),
            ):
                if field_val and issue_link not in field_val:
                    return f"{field_name} must include a full link to an issue (expected format: `{issue_link}1234` )"

        return None

    @classmethod
    def from_yaml(cls: type[QAIHMModelCodeGen], path: str | Path) -> QAIHMModelCodeGen:
        if not path or not os.path.exists(path):
            return QAIHMModelCodeGen()  # Default Schema
        return super().from_yaml(path)

    def to_model_yaml(self, model_id: str) -> Path:
        code_gen_path = QAIHM_MODELS_ROOT / model_id / "code-gen.yaml"
        self.to_yaml(code_gen_path, write_if_empty=False, delete_if_empty=True)
        return code_gen_path
