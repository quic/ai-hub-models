# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ruamel.yaml.scalarstring import LiteralScalarString

from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
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

    # The reason why various paths are disabled
    disabled_paths: ModelDisableReasonsMapping = field(
        default_factory=lambda: ModelDisableReasonsMapping()
    )

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

    # True for Collection model comprises of components, such as Whisper model's
    # encoder and decoder.
    is_collection_model: bool = False

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
    # Use LiteralScalarString so the YAML dump writes this on multiple lines instead of dumping '\n' directly
    additional_readme_section: LiteralScalarString = LiteralScalarString("")

    # If set, omits the "Example Usage" section from the HuggingFace readme.
    skip_example_usage: bool = False

    # By default inference tests are done using 8gen1 chipset to avoid overloading
    # newer devices. Some models don't work on 8gen1, so use 8gen3 for those.
    inference_on_8gen3: bool = False

    # TODO(#13765): remove this hack when we remove the quantized model folders
    # This folder is a "mirror" of the unquantized variant with the same name.
    # Evaluate and export scripts will call the script from the unquantized folder.
    quantized_mirror_of_model: Optional[str] = None

    # The model supports python versions that are at least this version. None == Any version
    python_version_greater_than_or_equal_to: Optional[str] = None
    python_version_greater_than_or_equal_to_reason: Optional[str] = None

    # The model supports python versions that are less than this version. None == Any version
    python_version_less_than: Optional[str] = None
    python_version_less_than_reason: Optional[str] = None

    def is_supported(self, precision: Precision, runtime: TargetRuntime) -> bool:
        """
        Return true if this precision + runtime combo is supported by this model.
        Return false if this model has a failure reason set for this runtime.
        """
        return not bool(self.failure_reason(precision, runtime))

    def failure_reason(
        self, precision: Precision, runtime: TargetRuntime
    ) -> Optional[str]:
        """
        Return the reason a model failed or None if the model did not fail.
        """
        if self.is_precompiled and runtime != TargetRuntime.QNN:
            return "Precompiled models are only supported via the QNN path."

        if precision and not runtime.supports_precision(precision):
            return f"{runtime} does not support precision {str(precision)}."

        if runtime == TargetRuntime.PRECOMPILED_QNN_ONNX:
            runtime = (
                TargetRuntime.QNN
            )  # QNN Support is a proxy for precompiled QNN ONNX.

        if reason := self.disabled_paths.get_disable_reasons(precision, runtime):
            if reason.has_failure:
                return reason.failure_reason

        return None

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
    def can_use_quantize_job(self) -> bool:
        """
        Whether the model can be quantized via quantize job.
        This may return true even if the model does list support for non-float precisions.
        """
        return (
            not self.is_precompiled
            and not self.is_aimet
            and not self.is_collection_model
        )

    @property
    def supports_quantization(self) -> bool:
        return any(x != Precision.float for x in self.supported_precisions)

    @property
    def default_precision(self) -> Precision:
        return self.supported_precisions[0]

    def validate(self) -> Optional[str]:
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
