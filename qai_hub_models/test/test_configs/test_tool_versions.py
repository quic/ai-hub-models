# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
import qai_hub as hub
from qai_hub.client import JobStatus, JobType
from qai_hub.public_api_pb2 import (
    CompileDetail,
    CompileJobResult,
    InferenceJobResult,
    JobResult,
    ProfileDetail,
    ProfileJobResult,
    ToolVersion,
)

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import InferenceEngine, QAIRTVersion, TargetRuntime

RESULTS_PATCH_TARGET = "qai_hub_models.configs.tool_versions.get_job_results"


def test_extract_tool_versions_from_compiled_model():
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QAIRT_SDK_VERSION: "2.28"},
        producer=MagicMock(spec=hub.CompileJob, _job_type=JobType.COMPILE),
    )
    assert ToolVersions.from_compiled_model(m) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )

    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
        producer=MagicMock(spec=hub.CompileJob, _job_type=JobType.COMPILE),
    )
    assert ToolVersions.from_compiled_model(m) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False)
    )

    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25"},
        producer=None,
    )
    with pytest.raises(ValueError):
        ToolVersions.from_compiled_model(m)


def test_extract_tool_versions_from_compile_job():
    def _make_results(versions: list[ToolVersion]):
        return JobResult(
            compile_job_result=CompileJobResult(
                compile_detail=CompileDetail(tool_versions=versions)
            )
        )

    # Compile job: Success
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
        producer=j,
    )
    j.get_target_model = lambda: m
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False)
    )

    # Compile job: Success with explicit qairt version in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
        options="--qairt_version latest",
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
        producer=j,
    )
    j.get_target_model = lambda: m
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
    )

    # Compile job: Failed with no options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
    )
    assert ToolVersions.from_job(j) == ToolVersions()
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions()

    # Compile job: Failed with target runtime QNN
    for rt in TargetRuntime:
        j = MagicMock(
            spec=hub.CompileJob,
            _job_type=JobType.COMPILE,
            get_status=lambda: JobStatus(JobStatus.State.FAILED),
            options=rt.aihub_target_runtime_flag,
        )
        if rt.inference_engine == InferenceEngine.QNN:
            assert ToolVersions.from_job(j) == ToolVersions(
                qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
            )
            assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
                qairt=QAIRTVersion.default()
            )
        else:
            assert ToolVersions.from_job(j) == ToolVersions()
            assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions()

    # Compile job: Failed with qairt version in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=2.28",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(job=j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )

    # Compile job: Failed with explicit qairt version tag in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=latest",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("latest", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion.latest()
    )

    # Compile job: Not QAIRT asset
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
        _owner=MagicMock(),
        job_id="0",
    )
    m = MagicMock(spec=hub.Model, producer=j, metadata={})
    j.get_target_model = lambda: m
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results([ToolVersion(name="ONNX", version="1.22.1")]),
    ):
        assert ToolVersions.from_job(j) == ToolVersions(onnx="1.22.1")


def test_extract_tool_versions_from_profile_job():
    def _make_results(versions: list[ToolVersion]):
        return JobResult(
            profile_job_result=ProfileJobResult(
                profile=ProfileDetail(tool_versions=versions)
            )
        )

    # Profile Job: Success
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results([ToolVersion(name="QAIRT", version="2.25.1234")]),
    ):
        assert ToolVersions.from_job(j) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False)
        )

    # Profile Job: Success (QNN instead of QAIRT)
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results(
            [
                ToolVersion(name="QNN", version="2.25.1234"),
                ToolVersion(name="ONNX Runtime", version="1.22.1"),
            ]
        ),
    ):
        assert ToolVersions.from_job(j) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
            onnx_runtime="1.22.1",
        )

    # Profile Job: Success (no QAIRT version in job results)
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(RESULTS_PATCH_TARGET, lambda *_: _make_results([])):
        assert ToolVersions.from_job(j) == ToolVersions()

    # Profile job: Failed with no options
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion.default()
    )

    # Profile job: Failed with qairt version in options
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=2.28",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(job=j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )

    # Profile job: Failed with explicit qairt version tag in options
    j = MagicMock(
        spec=hub.ProfileJob,
        _job_type=JobType.PROFILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=latest",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion.latest()
    )


def test_extract_tool_versions_from_inference_job():
    def _make_results(versions: list[ToolVersion]):
        return JobResult(
            inference_job_result=InferenceJobResult(
                detail=ProfileDetail(tool_versions=versions)
            )
        )

    # Inference job: Success
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results([ToolVersion(name="QAIRT", version="2.25.1234")]),
    ):
        assert ToolVersions.from_job(j) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False)
        )

    # Inference job: Success (QNN instead of QAIRT)
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results(
            [
                ToolVersion(name="QNN", version="2.25.1234"),
                ToolVersion(name="TensorFlow Lite", version="1.22.1"),
            ]
        ),
    ):
        assert ToolVersions.from_job(j) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
            tflite="1.22.1",
        )

    # Inference job: Success (no QAIRT version in job results)
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(RESULTS_PATCH_TARGET, lambda *_: _make_results([])):
        assert ToolVersions.from_job(j) == ToolVersions()

    # Inference job: Failed with no options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion.default()
    )

    # Inference job: Failed with qairt version in options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=2.28",
    )
    assert ToolVersions.from_job(job=j) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(job=j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )

    # Inference job: Failed with explicit qairt version tag in options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=latest",
    )
    assert ToolVersions.from_job(j) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions(
        qairt=QAIRTVersion.latest()
    )
