# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from unittest import mock

import pytest
from qai_hub import QuantizeDtype
from qai_hub.public_api_pb2 import Framework

from qai_hub_models.models.common import Precision, QAIRTVersion
from qai_hub_models.utils.base_config import BaseQAIHMConfig


@contextmanager
def reset_hub_frameworks_patches(
    frameworks, default_qaihm_version: str | None = None, api_url: str | None = None
):
    version_patch = (
        mock.patch(
            "qai_hub_models.models.common.QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION",
            default_qaihm_version,
        )
        if default_qaihm_version is not None
        else nullcontext()
    )
    api_url_patch = (
        mock.patch("qai_hub.hub._global_client.config.api_url", api_url)
        if api_url is not None
        else nullcontext()
    )
    with mock.patch(
        "qai_hub_models.models.common.get_framework_list",
        mock.MagicMock(return_value=mock.MagicMock(frameworks=frameworks)),
    ), mock.patch(
        "qai_hub_models.models.common.QAIRTVersion._FRAMEWORKS", dict()
    ), mock.patch(
        "qai_hub_models.models.common.QAIRTVersion._HUB_DEFAULT_FRAMEWORK", dict()
    ), mock.patch(
        "qai_hub_models.models.common.QAIRTVersion._QAIHM_DEFAULT_FRAMEWORK", dict()
    ), mock.patch(
        "qai_hub.hub._global_client.config.api_url", "https://app.aihub.qualcomm.com"
    ), version_patch, api_url_patch:
        yield


def test_precision_has_float():
    assert Precision.float.has_float_activations
    assert Precision.float.has_float_weights
    assert not Precision.w8a8.has_float_activations
    assert not Precision.w8a8.has_float_weights


def test_precision_eq():
    assert Precision.float == Precision.float
    assert Precision.float == Precision(None, None)
    assert Precision.float != Precision.w8a8
    assert Precision.w8a8 == Precision(QuantizeDtype.INT8, QuantizeDtype.INT8)
    assert Precision.w8a16 == Precision(QuantizeDtype.INT8, QuantizeDtype.INT16)
    assert Precision.w8a16 != Precision(QuantizeDtype.INT16, QuantizeDtype.INT8)
    assert Precision.float != 2


def test_precision_parse_serialize():
    assert str(Precision.float) == "float"
    assert str(Precision.w8a8) == "w8a8"
    assert str(Precision.w8a16) == "w8a16"

    assert Precision.parse("float") == Precision.float

    assert Precision.parse("w8a8") == Precision.w8a8
    assert Precision.parse("a8w8") == Precision.w8a8

    assert Precision.parse("w8a16") == Precision.w8a16
    assert Precision.parse("a16w8") == Precision.w8a16

    assert Precision.parse("w4") == Precision(QuantizeDtype.INT4, None)
    assert str(Precision.parse("w4")) == "w4"

    assert Precision.parse("a16") == Precision(None, QuantizeDtype.INT16)
    assert str(Precision.parse("a16")) == "a16"

    # Invalid bit width
    with pytest.raises(ValueError):
        Precision.parse("w8a24")


def test_qairt_version():
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
    with reset_hub_frameworks_patches(frameworks):
        # Get default from tag
        ai_hub_default = QAIRTVersion.default()
        assert ai_hub_default.tags == [QAIRTVersion.DEFAULT_AIHUB_TAG]
        assert ai_hub_default.hub_option == ""  # empty because it's the hub default
        assert ai_hub_default.explicit_hub_option == "--qairt_version 2.32"
        assert ai_hub_default == QAIRTVersion.DEFAULT_AIHUB_TAG
        assert ai_hub_default == ai_hub_default.api_version
        assert ai_hub_default == ai_hub_default.full_version
        assert ai_hub_default.is_default
        assert ai_hub_default.sdk_flavor is None

        # Get default using the api version
        aihub_default_api = QAIRTVersion(ai_hub_default.api_version)
        assert ai_hub_default == aihub_default_api
        assert aihub_default_api.is_default

        # Get default using the full api version
        aihub_default_fullapi = QAIRTVersion(ai_hub_default.full_version)
        assert ai_hub_default == aihub_default_fullapi
        assert aihub_default_fullapi.is_default
        assert aihub_default_fullapi.sdk_flavor is None
        assert aihub_default_fullapi != ai_hub_default.full_version + "-auto"

        # Get default using the full api version (auto flavor)
        aihub_default_fullapi_auto = QAIRTVersion(ai_hub_default.full_version + "-auto")
        assert ai_hub_default != aihub_default_fullapi_auto
        assert aihub_default_fullapi_auto.is_default
        assert aihub_default_fullapi_auto.sdk_flavor == "auto"
        assert aihub_default_fullapi != aihub_default_fullapi_auto
        assert aihub_default_fullapi_auto == ai_hub_default.full_version + "-auto"
        assert aihub_default_fullapi_auto != ai_hub_default.full_version

        # 0.0 does not exist, so it returns the default as a backup
        default_backup = QAIRTVersion("0.0", return_default_if_does_not_exist=True)
        assert ai_hub_default == default_backup
        assert default_backup.is_default

        # Latest
        latest = QAIRTVersion.latest()
        assert latest.tags == [QAIRTVersion.LATEST_AIHUB_TAG]
        assert latest.hub_option == f"--qairt_version {QAIRTVersion.LATEST_AIHUB_TAG}"
        assert latest.explicit_hub_option == "--qairt_version 2.33"

        # Untagged
        standard_version = QAIRTVersion("2.31")
        standard_version_2 = QAIRTVersion("2.31.0.25")
        assert standard_version == standard_version_2
        assert standard_version_2.api_version == standard_version.api_version
        assert standard_version.hub_option == "--qairt_version 2.31"
        assert standard_version.explicit_hub_option == "--qairt_version 2.31"

        # All Versions
        assert QAIRTVersion.all() == [QAIRTVersion(f.api_version) for f in frameworks]

        # Version that does not exist
        with pytest.raises(ValueError):
            QAIRTVersion("0.0")

        # Version that does not exist (no verification)
        assert (
            QAIRTVersion("1.1", validate_exists_on_ai_hub=False).full_version == "1.1"
        )

    # Verify QAIHM default behavior
    with reset_hub_frameworks_patches(frameworks, "2.33"):
        # Test default
        qaihm_default = QAIRTVersion.qaihm_default()
        assert qaihm_default.hub_option == "--qairt_version latest"
        assert qaihm_default.explicit_hub_option == "--qairt_version 2.33"
        assert qaihm_default != QAIRTVersion.default()
        assert not qaihm_default.is_default
        assert qaihm_default.is_qaihm_default
        assert not ai_hub_default.is_qaihm_default
        assert QAIRTVersion(QAIRTVersion.DEFAULT_QAIHM_TAG) == qaihm_default
        assert QAIRTVersion.DEFAULT_QAIHM_TAG in qaihm_default.tags

        # fallback is to the QAIHM default over the Hub default
        default_backup = QAIRTVersion("0.0", return_default_if_does_not_exist=True)
        assert qaihm_default == default_backup
        assert default_backup.is_qaihm_default

    # Verify QAIHM default behavior with same version as the AI Hub default
    with reset_hub_frameworks_patches(
        frameworks, default_qaihm_version=ai_hub_default.api_version
    ):
        qaihm_default = QAIRTVersion.qaihm_default()
        assert qaihm_default.hub_option == ""  # empty because it's the hub default
        assert qaihm_default.explicit_hub_option == "--qairt_version 2.32"
        assert qaihm_default == QAIRTVersion.default()
        assert QAIRTVersion.DEFAULT_AIHUB_TAG in qaihm_default.tags
        assert QAIRTVersion.DEFAULT_QAIHM_TAG in qaihm_default.tags

    # Verify QAIHM default behavior when the QAIHM default has no tags on Hub
    with reset_hub_frameworks_patches(frameworks, "2.31"):
        qaihm_default = QAIRTVersion.qaihm_default()
        assert qaihm_default.hub_option == "--qairt_version 2.31"
        assert qaihm_default.explicit_hub_option == "--qairt_version 2.31"

    # Verify "too old" QAIHM default behavior
    with reset_hub_frameworks_patches(frameworks, "2.30"):
        with pytest.raises(ValueError):
            QAIRTVersion.qaihm_default()
        with pytest.raises(ValueError):
            QAIRTVersion(QAIRTVersion.DEFAULT_QAIHM_TAG)

        # fallback is to the Hub default since the QAIHM default is not available
        default_backup = QAIRTVersion("0.0", return_default_if_does_not_exist=True)
        assert ai_hub_default == default_backup
        assert default_backup.is_default


def test_qairt_version_without_aihub_access():
    # Patch frameworks so this test continues to work regardless of AI Hub version changes.
    with reset_hub_frameworks_patches([], api_url=""):
        # Get default from tag
        ai_hub_default = QAIRTVersion.default()
        assert ai_hub_default.tags == [QAIRTVersion.DEFAULT_AIHUB_TAG]
        assert (
            str(ai_hub_default)
            == "QAIRT v0.0 | UNVERIFIED - NO AI HUB ACCESS | default"
        )

        # Get from full version
        full_version = QAIRTVersion("2.31.0.250130151446_114721-auto")
        assert full_version.api_version == "2.31"
        assert full_version.full_version == "2.31.0.250130151446_114721"
        assert full_version.sdk_flavor == "auto"

        # Untagged
        standard_version = QAIRTVersion("2.31")
        assert str(standard_version) == "QAIRT v2.31 | UNVERIFIED - NO AI HUB ACCESS"

        # If hub isn't available, any version is accepted, and the default backup is not used
        zero_version = QAIRTVersion("0.0", return_default_if_does_not_exist=True)
        zero_version_with_backup = QAIRTVersion(
            "0.0", return_default_if_does_not_exist=True
        )
        assert (
            str(zero_version)
            == str(zero_version_with_backup)
            == "QAIRT v0.0 | UNVERIFIED - NO AI HUB ACCESS"
        )

        # All Versions
        assert QAIRTVersion.all() == []


def test_qairt_version_pydantic_roundtrip():
    """
    Verifies that QAIRT versions are roundtripped to and from config files as-is,
    and aren't auto-resolved to current AI Hub QAIRT versions.
    """

    class VersionConfig(BaseQAIHMConfig):
        version: QAIRTVersion

    full_version = QAIRTVersion(
        "2.31.0.250130151446_114721-auto", validate_exists_on_ai_hub=False
    )
    assert (
        VersionConfig.model_validate_json(
            VersionConfig(version=full_version).model_dump_json()
        ).version
        == full_version
    )

    partial_version = QAIRTVersion("2.31.0-auto", validate_exists_on_ai_hub=False)
    assert partial_version.full_version_with_flavor == "2.31.0-auto"
    assert (
        VersionConfig.model_validate_json(
            VersionConfig(version=partial_version).model_dump_json()
        ).version
        == partial_version
    )
