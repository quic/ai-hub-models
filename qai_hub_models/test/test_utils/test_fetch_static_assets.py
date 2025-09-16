# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
from qai_hub.client import APIException

from qai_hub_models._version import __version__
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets
from qai_hub_models.utils.version_helpers import QAIHMVersion

MOBILENET_NAME = "MobileNet-v2"


def fetch_static_assets_internal_unavailable_patch():
    """
    Patches fetch_static_assets_internal so that it is not available (like in the public version of QAIHM).
    """
    return mock.patch(
        "qai_hub_models.utils.fetch_static_assets.fetch_static_assets_internal", None
    )


def hf_glob_patch(file_exists: bool = True, component_glob_result: list[str] = [""]):
    """
    Patches Hugging Face glob search to either return the set of file names passed to it if the file exists,
    or return None if the file does not exist.

    Parameters:
        file_exists:
            Whether the model should exist on Hugging Face.
        component_glob_result:
            When getting all components, glob will search for all component names via .*.
            This is the list of component names. The glob will return 1 file path per component.
    """

    def hf_glob(self, path: str):
        if file_exists and component_glob_result:
            # Act like the .* in the glob returns each component
            return [
                path.replace("*", f"_{component}" if component else "")
                for component in component_glob_result
            ]
        return [path] if file_exists else []

    return mock.patch("qai_hub_models.utils.huggingface.HfFileSystem.glob", hf_glob)


def hf_hub_download_patch():
    """
    Patches hf_hub_download to return the path the given args would be downloaded to, without actually downloading anything.
    """

    def hf_hub_download(repo_id: str, filename: str, local_dir: str, revision: str):
        return os.path.join(local_dir, filename)

    return mock.patch(
        "qai_hub_models.utils.huggingface.hf_hub_download", hf_hub_download
    )


def ai_hub_no_access_patch():
    """
    Patches the AI Hub client to act like the user does not have AI Hub access.
    """

    def get_frameworks():
        raise APIException("QAIHM API Access Failure Test")

    return mock.patch(
        "qai_hub_models.utils.qai_hub_helpers.hub.get_frameworks", get_frameworks
    )


def test_model_with_no_hf_assets():
    with hf_glob_patch(False), fetch_static_assets_internal_unavailable_patch():
        with pytest.raises(FileNotFoundError), TemporaryDirectory() as tmpdir:
            fetch_static_assets(
                "mobilenet_v2", TargetRuntime.ONNX, output_folder=tmpdir
            )


def test_hf_model_with_single_component():
    file_names = [
        f"{MOBILENET_NAME}_{Precision.float}.{TargetRuntime.ONNX.file_extension}"
    ]
    with TemporaryDirectory() as tmpdir:
        # Standard test
        with hf_glob_patch(), ai_hub_no_access_patch(), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2", TargetRuntime.ONNX, output_folder=tmpdir
            )
            assert len(local_paths) == 1
            assert len(urls) == 1
            assert str(object=local_paths[0]) == os.path.join(tmpdir, file_names[0])
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[0])
            assert __version__ in urls[0]

        # With download disabled, only URLs should be returned
        with hf_glob_patch(), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2", TargetRuntime.ONNX, output_folder=None
            )
            assert len(local_paths) == 0
            assert len(urls) == 1
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[0])
            assert __version__ in urls[0]

        # Changed version should be reflected in returned URL
        with hf_glob_patch(), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.ONNX,
                output_folder=None,
                qaihm_version_tag="v0.1",
            )
            assert len(local_paths) == 0
            assert len(urls) == 1
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[0])
            assert "v0.1" in urls[0]

        with hf_glob_patch(), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.ONNX,
                output_folder=None,
                qaihm_version_tag="latest",
            )
            assert len(local_paths) == 0
            assert len(urls) == 1
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[0])
            assert QAIHMVersion.latest_tag in urls[0]


def test_hf_model_with_multiple_components():
    component_names = ["Part1", "Part2"]
    file_names = [
        f"{MOBILENET_NAME}_Part1_{Precision.float}.{TargetRuntime.TFLITE.file_extension}",
        f"{MOBILENET_NAME}_Part2_{Precision.float}.{TargetRuntime.TFLITE.file_extension}",
    ]
    with TemporaryDirectory() as tmpdir:
        # Default (all components)
        with hf_glob_patch(
            component_glob_result=component_names
        ), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2", TargetRuntime.TFLITE, output_folder=tmpdir
            )
            assert len(local_paths) == 2
            assert len(urls) == 2
            assert str(local_paths[0]) == os.path.join(tmpdir, file_names[0])
            assert str(local_paths[1]) == os.path.join(tmpdir, file_names[1])
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[0])
            assert __version__ in urls[0]
            assert urls[1].startswith("https://huggingface.co")
            assert urls[1].endswith(file_names[1])
            assert __version__ in urls[1]

        # Search for specific component names (out of order)
        with hf_glob_patch(), hf_hub_download_patch():
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                output_folder=tmpdir,
                components=component_names[::-1],
            )
            assert len(local_paths) == 2
            assert len(urls) == 2
            assert str(local_paths[0]) == os.path.join(tmpdir, file_names[1])
            assert str(local_paths[1]) == os.path.join(tmpdir, file_names[0])
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(file_names[1])
            assert __version__ in urls[0]
            assert urls[1].startswith("https://huggingface.co")
            assert urls[1].endswith(file_names[0])
            assert __version__ in urls[1]
