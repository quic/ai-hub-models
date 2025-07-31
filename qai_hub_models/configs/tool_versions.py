# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import cast

import qai_hub as hub
from qai_hub.client import JobType
from qai_hub.public_rest_api import get_job_results

from qai_hub_models.models.common import (
    InferenceEngine,
    Optional,
    QAIRTVersion,
    TargetRuntime,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.qai_hub_helpers import extract_job_options


class ToolVersions(BaseQAIHMConfig):
    """
    Toolchains used to compile and profile models.

    BE CAREFUL when adding new tool versions.

    Equality of all fields in this class is used by scorecard to check whether a Hub
    compile job is the same as the previous week. Adding a very granular tool version
    (like AI Hub version) would break this without changing the equality check.
    """

    qairt: Optional[QAIRTVersion] = None
    onnx: Optional[str] = None
    onnx_runtime: Optional[str] = None
    tflite: Optional[str] = None

    @staticmethod
    def from_compiled_model(
        model: hub.Model,
    ) -> "ToolVersions":
        """
        Get the versions of tools used to compile this model.

        Parameters:
            model
                AI Hub model. Must be compiled by AI Hub.

        Returns:
            The tool versions required to compile this model.

        Raises:
            ValueError if the model was not compiled by AI Hub.
        """
        if model.producer is None or not model.producer._job_type == JobType.COMPILE:
            raise ValueError(
                "Model must be compiled with AI Hub to extract tool versions."
            )

        out = ToolVersions()

        # Get QAIRT Version from model metadata
        if qairt_version := model.metadata.get(hub.ModelMetadataKey.QAIRT_SDK_VERSION):
            out.qairt = QAIRTVersion(qairt_version, validate_exists_on_ai_hub=False)
        if qnn_version := model.metadata.get(hub.ModelMetadataKey.QNN_SDK_VERSION):
            out.qairt = QAIRTVersion(qnn_version, validate_exists_on_ai_hub=False)

        # Get TF Lite, ONNX versions from the compile job
        if out.qairt is None:  # we don't need to do this if it's a QAIRT-compiled model
            result = get_job_results(
                model.producer._owner.config, model.producer.job_id
            )
            for tool_version in result.compile_job_result.compile_detail.tool_versions:
                if tool_version.name == "ONNX":
                    out.onnx = tool_version.version

                if tool_version.name == "ONNX Runtime":
                    out.onnx_runtime = tool_version.version

                if tool_version.name == "TensorFlow Lite":
                    out.tflite = tool_version.version

        return out

    @staticmethod
    def from_job(job: hub.Job, parse_version_tags: bool = False) -> "ToolVersions":
        """
        Get the tool versions used for this job.
        For compile jobs, this is the toolchains used to compile the model.
        For profile / inference jobs, this is the toolchains used to run the model on device.

        Parameters:
            job
                AI Hub compile, profile, or inference job.

            parse_version_tags:
                When getting the QNN version from a failed AI Hub job, we rely on parsing the string options of the job.
                In this case, the version of often a tag, like 'latest' or 'default'. The definitions of these tags
                can change over time.

                If false, failed jobs that resolve to a QAIRT version tag are treated as if the QAIRT version can't
                be determined, and None is returned. This is the safest option and should be used in most situations.

                If true, version tags are parsed to match with their current meanings on AI Hub. BE CAUTIOUS USING THIS,
                as the QAIRT version represented by this tag may have changed since the job was submitted. Generally you
                should use this only if the job is recent enough that you know the current tags on AI Hub map to the same
                QAIRT versions when the job was submitted.

        Returns:
            The tool versions used to:
                Compile the model, if it's a compile job.
                Profile / Inference the model, if it's a profile / inference job.
                None if the version cannot be determined (the job failed and parse_version_tags is False).

            At the moment, only the QAIRT version is returned.

        Raises:
            ValueError if the job type is invalid.
        """
        # Use job_type instead of isinstance to support test mocking.
        if job._job_type not in [JobType.COMPILE, JobType.PROFILE, JobType.INFERENCE]:
            raise ValueError(
                f"Cannot extract QAIRT SDK version from job of type {job.job_type}"
            )

        if not job.get_status().success:
            # If the job is not successful, the only way to get the QAIRT version is to look at the job flags.
            job_options = extract_job_options(job)
            version: Optional[str] = None
            if "qairt_version" in job_options:
                version = cast(str, job_options["qairt_version"])
            elif "qnn_version" in job_options:
                version = cast(str, job_options["qnn_version"])
            else:
                if job._job_type == JobType.COMPILE:
                    # QAIRT is applicable for compile jobs only if the target runtime uses QAIRT converters.
                    if x := job_options.get("target_runtime"):
                        rts = [rt for rt in TargetRuntime if rt.value == x]
                        if (
                            len(rts) == 1
                            and rts[0].inference_engine == InferenceEngine.QNN
                        ):
                            version = "default"
                else:
                    version = "default"

            if version is None:
                return ToolVersions()

            if version in QAIRTVersion.all_tags():
                return ToolVersions(
                    qairt=QAIRTVersion(
                        version, validate_exists_on_ai_hub=parse_version_tags
                    )
                )

            return ToolVersions(
                qairt=QAIRTVersion(version, validate_exists_on_ai_hub=False)
            )

        if job._job_type == JobType.COMPILE:
            return ToolVersions.from_compiled_model(
                cast(hub.Model, cast(hub.CompileJob, job).get_target_model())
            )

        result = get_job_results(job._owner.config, job.job_id)
        if job._job_type == JobType.PROFILE:
            profile_detail = result.profile_job_result.profile
        elif job._job_type == JobType.INFERENCE:
            profile_detail = result.inference_job_result.detail
        else:
            # This is unreachable, but we write it for type checking.
            assert False

        out = ToolVersions()
        for tool_version in profile_detail.tool_versions:
            if tool_version.name == "QAIRT" or tool_version.name == "QNN":
                out.qairt = QAIRTVersion(
                    tool_version.version, validate_exists_on_ai_hub=False
                )
            if tool_version.name == "ONNX Runtime":
                out.onnx_runtime = tool_version.version

            if tool_version.name == "TensorFlow Lite":
                out.tflite = tool_version.version

        return out
