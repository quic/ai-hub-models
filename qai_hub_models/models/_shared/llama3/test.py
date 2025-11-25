# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import qai_hub as hub

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.model import (
    LLM_AIMETOnnx,
    LLMBase,
    cleanup,
)
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import ONNXBundle
from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.llm_helpers import copy_qairt_files_for_genie_bundle
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.testing import patch_qai_hub


def complete_genie_bundle_and_run_on_device(
    device: ScorecardDevice, genie_bundle_path: str
) -> None:
    copy_qairt_files_for_genie_bundle(
        device.reference_device,
        Path(genie_bundle_path),
        os.environ["QAIRT_SDK_PATH"],
    )

    # Add prompt.txt to genie_bundle
    with open(os.path.join(genie_bundle_path, "prompt.txt"), "w") as f:
        f.write(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant. Be concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    # Run QDC APIs to validate the bundle on-device.
    from qai_hub_models.utils.qdc.qdc_jobs import submit_genie_bundle_to_qdc_device

    _, avg_tps, min_ttft = submit_genie_bundle_to_qdc_device(
        os.environ["QDC_API_TOKEN"], device.reference_device.name, genie_bundle_path
    )

    # Cleanup the generated genie bundle after test
    shutil.rmtree(genie_bundle_path)
    return avg_tps, min_ttft


def _mock_from_pretrained(model_cls, context_length: int, sequence_length: int):
    model = MagicMock()
    model.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def from_bundle_path_patch(bundle_path: str | os.PathLike) -> ONNXBundle:
    return ONNXBundle(
        bundle_path=Path(bundle_path),
        onnx_graph_name="model.onnx",
        aimet_encodings_name="model.encodings",
    )


def split_onnx_patch(*args, num_splits, **kwargs):
    return [from_bundle_path_patch(f"{i}") for i in range(num_splits)]


# reusable patching function
def _create_patches(
    model_cls: type[LLM_AIMETOnnx],
    base_name: str,
    context_length: int,
    sequence_length: int,
    tmp_path: Path,
):
    mock_from_pretrained = _mock_from_pretrained(
        model_cls, context_length, sequence_length
    )
    mock_from_pretrained.return_value.sample_inputs.return_value = {
        "input0": [np.array([1.0, 2.0])]
    }
    # patching to not download from huggingface.
    patch_model = patch(
        f"qai_hub_models.models.{base_name}.Model.from_pretrained",
        mock_from_pretrained,
    )

    patch_fp_model = patch(
        f"qai_hub_models.models.{base_name}.FP_Model.from_pretrained",
        return_value=Mock(),
    )

    patch_onnx_checker = patch("onnx.checker.check_model")

    patch_split_onnx = patch(
        "qai_hub_models.models._shared.llm.split_onnx_utils.utils.split_onnx",
        side_effect=split_onnx_patch,
    )

    patch_onnx_files = patch.object(
        ONNXBundle, "from_bundle_path", side_effect=from_bundle_path_patch
    )

    patch_get_or_create_cached_model = patch(
        "qai_hub_models.models._shared.llm.export.get_or_create_cached_model",
        return_value=Mock(),
    )
    patch_tool_versions = patch(
        "qai_hub_models.configs.tool_versions.ToolVersions.from_job",
        return_value=ToolVersions(
            qairt=QAIRTVersion("2.34", validate_exists_on_ai_hub=False)
        ),
    )

    return (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    )


def test_cli_device_with_skips_unsupported_precision_device(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
):
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4A16",
        ]

        with pytest.raises(
            ValueError,
            match=r"The selected precision \(w4a16\) is not supported on this target device",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips_unsupported_context_length(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
):
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4",
        ]

        with pytest.raises(
            ValueError,
            match=r"The llama_v3_2_3b_instruct's context length is too large to deploy on SA8295P\. Please set the context length to 1024 or lower\.",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
):
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
        ]
        if not skip_inferencing:
            sys.argv.append("--do-inferencing")
        if skip_profiling:
            sys.argv.append("--skip-profiling")

        export_main()  # Call the main function to submit the compile jobs

        # Compile is called parts * 2 times (num_splits token parts, num_splits prompt parts)
        assert mock_hub.submit_compile_job.call_count == parts * 2
        call_args_list = mock_hub.submit_compile_job.call_args_list

        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts

        call_args_list = mock_hub.submit_link_job.call_args_list

        assert all(
            call.kwargs["name"] == f"{base_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(call_args_list)
        )

        # Skip profile and inference combinations.
        if skip_inferencing:
            mock_hub.submit_inference_job.assert_not_called()
        else:
            mock_hub.submit_inference_job.assert_called()

        if skip_profiling:
            mock_hub.submit_profile_job.assert_not_called()
        else:
            mock_hub.submit_profile_job.assert_called()
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_chipset_with_options(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    chipset: str,
    context_length: int,
    sequence_length: int,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
):
    (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        compile_options = "compile_extra"
        profile_options = "profile_extra"
        link_options = "link_extra"

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",  # script name
            "--chipset",
            chipset,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--compile-options",
            compile_options,
            "--profile-options",
            profile_options,
            "--link-options",
            link_options,
            "--sequence-length",
            str(sequence_length),
            "--context-length",
            str(context_length),
            "--target-runtime",
            target_runtime.value,
            "--do-inferencing",
        ]

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_profile_job.call_count == parts * 2
        assert mock_hub.submit_inference_job.call_count == parts * 2

        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            assert all(
                f"chipset:{chipset}" in call.kwargs["device"].attributes
                for call in mock.call_args_list
            )

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts
        assert all(
            call.kwargs["name"] == f"{base_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )
        mock_get_hub_link_options = (
            mock_from_pretrained.return_value.get_hub_link_options
        )
        assert mock_get_hub_link_options.call_count == parts
        assert all(
            call.args == (target_runtime, link_options)
            for call in mock_get_hub_link_options.call_args_list
        )

        mock_get_hub_compile_options = (
            mock_from_pretrained.return_value.get_hub_compile_options
        )

        assert mock_get_hub_compile_options.call_count == parts * 2
        for call in mock_get_hub_compile_options.call_args_list:
            assert call.args == (target_runtime, precision, compile_options)
            assert "context_graph_name" in call.kwargs
            assert (
                call.kwargs["context_graph_name"]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qairt_context_graph_name._mock_name
            )

        # Profile parts * 2 times
        assert mock_hub.submit_profile_job.call_count == parts * 2
        mock_get_hub_profile_options = (
            mock_from_pretrained.return_value.get_hub_profile_options
        )
        assert mock_get_hub_profile_options.call_count == parts * 2
        for call in mock_get_hub_profile_options.call_args_list:
            assert len(call.args) == 3
            assert call.args[:2] == (target_runtime, profile_options)
            assert (
                call.args[2]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qairt_context_graph_name._mock_name
            )

        assert mock_hub.submit_inference_job.call_count == parts * 2

        assert tmp_path.exists()
        assert tmp_path.is_dir()

        # TODO (#15224): Remove from_pretrained as part of inference?
        assert mock_from_pretrained.call_count == 4
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[1].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["sequence_length"]
            == sequence_length
        )
        # In instantiations list (160) from _shared/llm/export.py
        assert mock_from_pretrained.call_args_list[1].kwargs["sequence_length"] == 1


# for llama3 all components are tested, i.e. no option to select individual components 'part_1_of_5', 'part_2_of_5', 'part_3_of_5', 'part_4_of_5', 'part_5_of_5'
def test_cli_default_device_select_component(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
):
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    patch_torch_inference = patch("qai_hub_models.utils.compare.torch_inference")

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model as mock_get_or_create_cached_model,
        patch_torch_inference as mock_torch_inference,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        mock_torch_inference.return_value = [
            np.array([1.0, 2.0])
        ]  # return mock value for torch inference.

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--model-cache-mode",
            str(cache_mode.name.lower()),
            "--device",
            device.name,
            "--target-runtime",
            target_runtime.value,
        ]
        if skip_download:
            sys.argv.append("--skip-downloading")
        if skip_summary:
            sys.argv.append("--skip-summary")

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_link_job.call_count == parts
        assert mock_hub.submit_profile_job.call_count == parts * 2

        # Check names
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
        ]:
            for i, call in enumerate(mock.call_args_list):
                model_type = "prompt" if i < parts else "token"
                assert (
                    call.kwargs["name"]
                    == f"{base_name}_{model_type}_{(i % parts) + 1}_of_{parts}"
                )

        assert mock_get_or_create_cached_model.call_count == parts * 2
        # Check cache mode is correct.
        for call in mock_get_or_create_cached_model.call_args_list:
            assert call.kwargs["model_name"] == base_name
            assert call.kwargs["cache_mode"] == cache_mode

        # check compile jobs have correct device name.
        assert all(
            call.kwargs["device"].name == device.name
            for call in mock_hub.submit_compile_job.call_args_list
        )

        assert tmp_path.exists()
        assert tmp_path.is_dir()


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (Path(output_path) / "model_seqlen1_cl4096.onnx").exists()
        and (Path(output_path) / "model_seqlen128_cl4096.onnx").exists()
    ):
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            allow_cpu_to_quantize=True,
            checkpoint=checkpoint,
            num_samples=num_samples,
        )
        cleanup()
    return output_path
