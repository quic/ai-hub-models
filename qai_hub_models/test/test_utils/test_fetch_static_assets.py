# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
import qai_hub as hub
from qai_hub.client import APIException

from qai_hub_models._version import __version__
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets
from qai_hub_models.utils.version_helpers import QAIHMVersion

MOBILENET_NAME = "MobileNet-v2"


def fetch_static_assets_internal_unavailable_patch():
    """Patches fetch_static_assets_internal so that it is not available (like in the public version of QAIHM)."""
    return mock.patch(
        "qai_hub_models.utils.fetch_static_assets.fetch_static_assets_internal", None
    )


def hf_glob_patch(
    file_exists: bool = True, component_glob_result: list[str] | None = None
):
    """
    Patches Hugging Face glob search to either return the set of file names passed to it if the file exists,
    or return None if the file does not exist.

    Parameters
    ----------
        file_exists:
            Whether the model should exist on Hugging Face.
        component_glob_result:
            When getting all components, glob will search for all component names via .*.
            This is the list of component names. The glob will return 1 file path per component.
    """
    if component_glob_result is None:
        component_glob_result = [""]

    def hf_glob(self, path: str, revision: str | None = None):
        if revision is not None:
            org, repo, file = path.split("/", maxsplit=2)
            path = "/".join([org, repo + f"@{revision}", file])

        if file_exists and component_glob_result:
            # Act like the .* in the glob returns each component
            return [
                path.replace("*", f"_{component}" if component else "")
                for component in component_glob_result
            ]

        return [path] if file_exists else []

    return mock.patch("qai_hub_models.utils.huggingface.HfFileSystem.glob", hf_glob)


def hf_hub_download_patch():
    """Patches hf_hub_download to return the path the given args would be downloaded to, without actually downloading anything."""

    def hf_hub_download(repo_id: str, filename: str, local_dir: str, revision: str):
        return os.path.join(local_dir, filename)

    return mock.patch(
        "qai_hub_models.utils.huggingface.hf_hub_download", hf_hub_download
    )


def ai_hub_no_access_patch():
    """Patches the AI Hub Workbench client to act like the user does not have AI Hub Workbench access."""

    def get_frameworks():
        raise APIException("QAIHM API Access Failure Test")

    return mock.patch(
        "qai_hub_models.utils.qai_hub_helpers.hub.get_frameworks", get_frameworks
    )


def download_file_patch(
    file_contents: Iterable[str | BaseQAIHMConfig | None],
) -> mock._patch:
    """
    Patches AI Hub Models' download file API to return set file contents, rather than actually fetching from the internet.

    Parameters
    ----------
    file_contents
        A list of file contents to save. Call idx i of download_file() will return a path to a file with the contents of this list at index i.

        If the file contents are 'None', raises a ValueError to simulate a download failure.

    Returns
    -------
    mock._patch
        Test Patch
    """
    files_iter = iter(file_contents)

    def download_file(web_url: str, dst_path: str, *args, **kwargs):
        if not files_iter:
            raise ValueError(
                "TESTING: Ran out of simulated file contents for download_file to return."
            )

        file = next(files_iter)
        if file is None:
            raise ValueError("TESTING: Simulated download error.")
        if isinstance(file, BaseQAIHMConfig):
            file.to_yaml(dst_path)
        if isinstance(file, str):
            with open(dst_path, "w") as dstf:
                dstf.write(dst_path)

        return dst_path

    return mock.patch(
        "qai_hub_models.utils.fetch_static_assets.download_file", download_file
    )


def test_model_with_no_hf_assets():
    with (
        hf_glob_patch(False),
        fetch_static_assets_internal_unavailable_patch(),
        pytest.raises(FileNotFoundError),
        TemporaryDirectory() as tmpdir,
    ):
        fetch_static_assets("mobilenet_v2", TargetRuntime.ONNX, output_folder=tmpdir)


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

        # Version of info.yaml to return for previous releases.
        info_yaml = QAIHMModelInfo.from_model("mobilenet_v2")

        # Changed version should be reflected in returned URL
        with hf_glob_patch(), hf_hub_download_patch(), download_file_patch([info_yaml]):
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

        with hf_glob_patch(), hf_hub_download_patch(), download_file_patch([info_yaml]):
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

        # Throw if the old info.yaml is not available
        with (
            pytest.raises(
                FileNotFoundError, match=r"Unable to find asset data for mobilenet_v2"
            ),
            download_file_patch([None]),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.ONNX,
                output_folder=None,
                qaihm_version_tag="v0.1",
            )

        # Throw if the old info.yaml is not parseable
        with (
            pytest.raises(
                FileNotFoundError, match=r"Unable to parse asset data for mobilenet_v2"
            ),
            download_file_patch([""]),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.ONNX,
                output_folder=None,
                qaihm_version_tag="v0.1",
            )


def test_hf_model_with_multiple_components():
    component_names = ["Part1", "Part2"]
    file_names = [
        f"{MOBILENET_NAME}_Part1_{Precision.float}.{TargetRuntime.TFLITE.file_extension}",
        f"{MOBILENET_NAME}_Part2_{Precision.float}.{TargetRuntime.TFLITE.file_extension}",
    ]
    with TemporaryDirectory() as tmpdir:
        # Default (all components)
        with (
            hf_glob_patch(component_glob_result=component_names),
            hf_hub_download_patch(),
        ):
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


def test_hf_model_with_device_specific_assets():
    expected_huggingface_file_names = [
        f"precompiled/qualcomm-snapdragon-x-elite/{MOBILENET_NAME}_{Precision.float}.{TargetRuntime.PRECOMPILED_QNN_ONNX.file_extension}"
    ]
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(
            ValueError,
            match=r"You must specify a device to fetch an asset that is device-specific.",
        ):
            local_paths, urls = fetch_static_assets(
                "mobilenet_v2", TargetRuntime.PRECOMPILED_QNN_ONNX, output_folder=tmpdir
            )

        with hf_glob_patch(), ai_hub_no_access_patch(), hf_hub_download_patch():
            for device in [
                hub.Device("Snapdragon X Elite CRD"),
                hub.Device(attributes="chipset:qualcomm-snapdragon-x-elite"),
            ]:
                local_paths, urls = fetch_static_assets(
                    "mobilenet_v2",
                    TargetRuntime.PRECOMPILED_QNN_ONNX,
                    device=device,
                    output_folder=tmpdir,
                )
                assert len(local_paths) == 1
                assert len(urls) == 1
                assert str(object=local_paths[0]) == os.path.join(
                    tmpdir, expected_huggingface_file_names[0]
                )
                assert urls[0].startswith("https://huggingface.co")
                assert urls[0].endswith(expected_huggingface_file_names[0])
                assert __version__ in urls[0]

            # Verify we can fetch from an older version of QAIHM.
            info_yaml = QAIHMModelInfo.from_model("mobilenet_v2")
            devices_yaml = DevicesAndChipsetsYaml.load()
            with download_file_patch([info_yaml, devices_yaml]):
                local_paths, urls = fetch_static_assets(
                    "mobilenet_v2",
                    TargetRuntime.PRECOMPILED_QNN_ONNX,
                    device=hub.Device(attributes="chipset:qualcomm-snapdragon-x-elite"),
                    output_folder=tmpdir,
                    qaihm_version_tag="0.92",
                )
                assert len(local_paths) == 1
                assert len(urls) == 1
                assert str(object=local_paths[0]) == os.path.join(
                    tmpdir, expected_huggingface_file_names[0]
                )
                assert urls[0].startswith("https://huggingface.co")
                assert urls[0].endswith(expected_huggingface_file_names[0])
                assert "0.92" in urls[0]

        ##
        # Verify we fail correctly if we fetch from a bad older version of QAIHM.
        ##

        # Throw if the old chipsets.yaml is not available
        with (
            pytest.raises(
                FileNotFoundError,
                match=r"Unable to find supported devices and chipsets for QAIHM",
            ),
            download_file_patch([info_yaml, None]),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                device=hub.Device(attributes="chipset:qualcomm-snapdragon-x-elite"),
                output_folder=tmpdir,
                qaihm_version_tag="0.92",
            )

        # Throw if the old chipsets.yaml is not parseable
        with (
            download_file_patch([info_yaml, ""]),
            pytest.raises(
                FileNotFoundError,
                match=r"Unable to parse supported devices and chipsets for QAIHM",
            ),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                device=hub.Device(attributes="chipset:qualcomm-snapdragon-x-elite"),
                output_folder=tmpdir,
                qaihm_version_tag="0.92",
            )
