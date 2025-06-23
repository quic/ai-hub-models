# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import sys
from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.testing import patch_qai_hub


def _mock_from_pretrained(model_cls, context_length, sequence_length):
    model = MagicMock()
    model.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def _mock_glob(path):
    return ["path/to/name" + os.path.splitext(path)[1]]


# reusable patching function
def _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path):
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
    )  # type: ignore

    patch_fp_model = patch(
        f"qai_hub_models.models.{base_name}.FP_Model.from_pretrained",
        return_value=Mock(),
    )  # type: ignore

    patch_glob = patch("glob.glob", side_effect=_mock_glob)
    patch_onnx_checker = patch("onnx.checker.check_model")
    patch_split_onnx = patch(
        "qai_hub_models.models._shared.llama3.split_onnx_utils.utils.split_onnx"
    )
    patch_onnx_files = patch.object(
        Path, "glob", return_value=[Path(tmp_path / "onnx_file.onnx")]
    )
    patch_get_or_create_cached_model = patch(
        "qai_hub_models.models._shared.llama3.export.get_or_create_cached_model",
        return_value=Mock(),
    )

    return (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    )


def test_cli_device_with_skips_unsupported_device(
    export_main, model_cls, tmp_path, base_name
):
    (
        _,
        patch_model,
        patch_fp_model,
        patch_glob,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    with (
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ):

        sys.argv = [
            "export.py",
            "--device",
            "Google Pixel 4",
            "--skip-inferencing",
            "--skip-profiling",
            "--output-dir",
            tmp_path.name,
        ]

        with pytest.raises(ValueError) as e:
            export_main()  # Call the main function to submit the compile jobs
        assert (
            str(e.value)
            == "The selected device does not support weight sharing. This script relies on weight sharing and can only target devices that support it (Snapdragon 8 Gen 2 and later)."
        )


def test_cli_device_with_skips(
    export_main,
    model_cls,
    tmp_path,
    base_name,
    parts,
    device,
    skip_inferencing,
    skip_profiling,
):
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            tmp_path.name,
        ]
        if skip_inferencing:
            sys.argv.append("--skip-inferencing")
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
    export_main,
    model_cls,
    tmp_path,
    base_name,
    parts,
    chipset,
    context_length,
    sequence_length,
):
    context_length = 4096
    sequence_length = 128
    (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker as mock_onnx_checker,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        compile_options = "compile_extra"
        profile_options = "profile_extra"

        sys.argv = [
            "export.py",  # script name
            "--chipset",
            # "qualcomm-snapdragon-8gen2",
            chipset,
            "--output-dir",
            tmp_path.name,
            "--target-runtime",
            "qnn_context_binary",
            "--compile-options",
            compile_options,
            "--profile-options",
            profile_options,
            "--sequence-length",
            str(sequence_length),
            "--context-length",
            str(context_length),
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

        mock_get_hub_compile_options = (
            mock_from_pretrained.return_value.get_hub_compile_options
        )

        # TODO (#15223): Wrong precision call(<TargetRuntime.QNN_CONTEXT_BINARY: 'qnn_context_binary'>, w8a16, 'compile_extra'): Should be w4a16
        assert mock_get_hub_compile_options.call_count == parts * 2
        assert all(
            call.args
            == (TargetRuntime.QNN_CONTEXT_BINARY, Precision.w8a16, compile_options)
            for call in mock_get_hub_compile_options.call_args_list
        )

        # Profile parts * 2 times
        assert mock_hub.submit_profile_job.call_count == parts * 2

        mock_get_hub_profile_options = (
            mock_from_pretrained.return_value.get_hub_profile_options
        )
        assert mock_get_hub_profile_options.call_count == 2

        assert all(
            call.args == (TargetRuntime.QNN_CONTEXT_BINARY, profile_options)
            for call in mock_get_hub_profile_options.call_args_list
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
        # In instantiations list (160) from _shared/llama3/export.py
        assert mock_from_pretrained.call_args_list[1].kwargs["sequence_length"] == 1


# for llama3 all components are tested, i.e. no option to select individual components 'part_1_of_5', 'part_2_of_5', 'part_3_of_5', 'part_4_of_5', 'part_5_of_5'
def test_cli_default_device_select_component(
    export_main,
    model_cls,
    tmp_path,
    base_name,
    parts,
    device,
    cache_mode,
    skip_download,
    skip_summary,
):
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    patch_torch_inference = patch("qai_hub_models.utils.compare.torch_inference")

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_glob,
        patch_onnx_checker as mock_onnx_checker,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model as mock_get_or_create_cached_model,
        patch_torch_inference as mock_torch_inference,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        mock_torch_inference.return_value = [
            np.array([1.0, 2.0])
        ]  # return mock value for torch inference.

        sys.argv = [
            "export.py",
            "--output-dir",
            tmp_path.name,
            "--model-cache-mode",
            str(cache_mode.name.lower()),
            "--device",
            device.name,
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
        assert mock_hub.submit_inference_job.call_count == parts * 2

        # Check names
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
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

        # check compil jobs have correct device name.
        assert all(
            call.kwargs["device"].name == device.name
            for call in mock_hub.submit_compile_job.call_args_list
        )

        assert tmp_path.exists()
        assert tmp_path.is_dir()
