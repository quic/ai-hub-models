# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub.public_api_pb2 import Framework

from qai_hub_models.models.common import QAIRTVersion
from qai_hub_models.scorecard.path_compile import (
    QAIRTVersionEnvvar,
    ScorecardCompilePath,
    TargetRuntime,
)
from qai_hub_models.test.test_models.test_common import reset_hub_frameworks_patches
from qai_hub_models.test.utils.set_env import set_temp_env


def test_compile_qnn_version():
    """
    This verifies behavior of ScorecardCompilePath.get_compile_options() with
    different combinations of:
      * default AI Hub QAIRT version
      * default AI Hub models QAIRT version
      * QAIRT version override environment variables in the test environment
    """
    # Patch frameworks so this test continues to work regardless of AI Hub version changes.
    frameworks = [
        Framework(
            name="QAIRT",
            api_tags=[],
            api_version="2.31",
            full_version="2.31.0.250130151446_114721",
        ),
        Framework(
            name="QAIRT",
            api_tags=[QAIRTVersion.DEFAULT_AIHUB_TAG],
            api_version="2.32",
            full_version="2.32.6.250402152434_116405",
        ),
        Framework(
            name="QAIRT",
            api_tags=[QAIRTVersion.LATEST_AIHUB_TAG],
            api_version="2.33",
            full_version="2.33.0.250327124043_117917",
        ),
    ]

    # Test working AI Hub instance
    for api_version in (None, "2.32", "2.31"):
        with reset_hub_frameworks_patches(frameworks, api_version):
            qairt_agnostic_compile_path = ScorecardCompilePath.TFLITE
            assert (
                not qairt_agnostic_compile_path.runtime.qairt_version_changes_compilation
            )

            qairt_dependent_compile_path = ScorecardCompilePath.QNN_CONTEXT_BINARY
            assert (
                qairt_dependent_compile_path.runtime.qairt_version_changes_compilation
            )

            # fmt: off
            # PRECOMPILED_QNN_ONNX has a special QAIRT version that differs from AI Hub Models' default.
            qairt_dependent_compile_path_with_different_qairt_version = ScorecardCompilePath.PRECOMPILED_QNN_ONNX
            assert qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation
            precompiled_qnn_onnx_flag = TargetRuntime.PRECOMPILED_QNN_ONNX.default_qairt_version.explicit_hub_option

            if api_version is not None:
                default_qaihm_qnn_flag = QAIRTVersion.qaihm_default().explicit_hub_option

                # QAIRT version set to QAIHM default
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: QAIRTVersion.DEFAULT_QAIHM_TAG}):
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path.get_compile_options()
                    assert default_qaihm_qnn_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                    assert precompiled_qnn_onnx_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # No flag set (same behavior as if flag was the same as the default AI Hub Models qQAIRT version)
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: None}):
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path.get_compile_options()
                    assert default_qaihm_qnn_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                    assert precompiled_qnn_onnx_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: "2.31"}):
                    override_qairt_flag = QAIRTVersion("2.31").explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag
                # It is still explicit even if it's the default AI Hub version.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: QAIRTVersion.DEFAULT_AIHUB_TAG}):
                    override_qairt_flag = QAIRTVersion(QAIRTVersion.DEFAULT_AIHUB_TAG).explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
                # It is still explicit even if the selected API version is the same as the AI Hub Models default tag.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: QAIRTVersion.qaihm_default().api_version}):
                    override_qairt_flag = QAIRTVersion.qaihm_default().explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
            # fmt: on
