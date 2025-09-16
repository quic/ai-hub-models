# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub.public_api_pb2 import Framework

from qai_hub_models.models.common import InferenceEngine, QAIRTVersion
from qai_hub_models.scorecard.path_compile import (
    QAIRTVersionEnvvar,
    ScorecardCompilePath,
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
            api_tags=[],
            api_version="2.33",
            full_version="2.33.0.250327124043_117917",
        ),
        Framework(
            name="QAIRT",
            api_tags=[QAIRTVersion.LATEST_AIHUB_TAG],
            api_version="2.36",
            full_version="2.36.0.250327124043_117917",
        ),
    ]

    # Test working AI Hub instance
    for api_version, per_runtime_api_version in (
        ("2.32", {InferenceEngine.ONNX: "2.33"}),
        ("2.31", {InferenceEngine.ONNX: "2.32"}),
    ):
        with reset_hub_frameworks_patches(
            frameworks, api_version, per_runtime_api_version
        ):
            qairt_agnostic_compile_path = ScorecardCompilePath.TFLITE
            assert (
                not qairt_agnostic_compile_path.runtime.qairt_version_changes_compilation
            )

            qairt_dependent_compile_path = ScorecardCompilePath.QNN_CONTEXT_BINARY
            assert (
                qairt_dependent_compile_path.runtime.qairt_version_changes_compilation
            )

            # fmt: off
            # Spome paths have a special QAIRT version that differs from AI Hub Models' default.
            paths_with_different_qairt_version : list[tuple[ScorecardCompilePath, str]] = []
            for profile_path_with_different_qairt_version in ScorecardCompilePath:
                if profile_path_with_different_qairt_version.runtime.inference_engine not in per_runtime_api_version:
                    continue

                assert not profile_path_with_different_qairt_version.runtime.is_aot_compiled or profile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation
                path_flag_with_different_qairt_version = profile_path_with_different_qairt_version.runtime.default_qairt_version.explicit_hub_option
                paths_with_different_qairt_version.append((profile_path_with_different_qairt_version, path_flag_with_different_qairt_version))

            if api_version is not None:
                default_qaihm_qnn_flag = qairt_agnostic_compile_path.runtime.default_qairt_version.explicit_hub_option

                # QAIRT version set to QAIHM default
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: QAIRTVersionEnvvar.default()}):
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path.get_compile_options()
                    assert default_qaihm_qnn_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    for qairt_dependent_compile_path_with_different_qairt_version, path_flag_with_different_qairt_version in paths_with_different_qairt_version:
                        assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                        if qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation:
                            assert path_flag_with_different_qairt_version in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
                        else:
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # No flag set (same behavior as if flag was the same as the default AI Hub Models qQAIRT version)
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: None}):
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path.get_compile_options()
                    assert default_qaihm_qnn_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    for qairt_dependent_compile_path_with_different_qairt_version, path_flag_with_different_qairt_version in paths_with_different_qairt_version:
                        if qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation:
                            assert path_flag_with_different_qairt_version in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
                        else:
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: "2.31"}):
                    override_qairt_flag = QAIRTVersion("2.31").explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    for qairt_dependent_compile_path_with_different_qairt_version, _ in paths_with_different_qairt_version:
                        if qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation:
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
                        else:
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag
                # It is still explicit even if it's the default AI Hub version.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: QAIRTVersion.DEFAULT_AIHUB_TAG}):
                    override_qairt_flag = QAIRTVersion(QAIRTVersion.DEFAULT_AIHUB_TAG).explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    for qairt_dependent_compile_path_with_different_qairt_version, _ in paths_with_different_qairt_version:
                        if qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation:
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
                        else:
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)

                # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
                # It is still explicit even if the selected API version is the same as the AI Hub Models default tag.
                with set_temp_env({QAIRTVersionEnvvar.VARNAME: qairt_agnostic_compile_path.runtime.default_qairt_version.api_version}):
                    override_qairt_flag = qairt_agnostic_compile_path.runtime.default_qairt_version.explicit_hub_option
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options()
                    assert QAIRTVersion.HUB_FLAG not in qairt_agnostic_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options()
                    assert override_qairt_flag in qairt_dependent_compile_path.get_compile_options(include_default_qaihm_qnn_version=True)
                    for qairt_dependent_compile_path_with_different_qairt_version, _ in paths_with_different_qairt_version:
                        if qairt_dependent_compile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation:
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert override_qairt_flag in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
                        else:
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options()
                            assert QAIRTVersion.HUB_FLAG not in qairt_dependent_compile_path_with_different_qairt_version.get_compile_options(include_default_qaihm_qnn_version=True)
            # fmt: on
