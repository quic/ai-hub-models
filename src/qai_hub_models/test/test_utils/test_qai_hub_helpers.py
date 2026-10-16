# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import qai_hub as hub
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models import Precision
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.job_options import extract_job_options
from qai_hub_models.utils.qai_hub_helpers import (
    assert_success_and_get_target_models,
    raise_if_fp_is_unsupported,
)


def assert_options_eq(options: str, options_dict: dict[str, str | bool]) -> None:
    assert extract_job_options(MagicMock(spec=hub.Job, options=options)) == options_dict


def test_extract_job_options() -> None:
    assert_options_eq("", {})
    assert_options_eq("--boolean_flag", {"boolean_flag": True})
    assert_options_eq(
        "--blah text --boolean_flag", {"blah": "text", "boolean_flag": True}
    )
    assert_options_eq(
        "--boolean_flag --dict-input='blah=true;x=y'",
        {"dict-input": "blah=true;x=y", "boolean_flag": True},
    )
    assert_options_eq("--dict-input 'blah=true;x=y'", {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input "blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input="blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq("--dict-input=blah=true;x=y", {"dict-input": "blah=true;x=y"})


# ---------------------------------------------------------------------------
# raise_if_fp_is_unsupported
# ---------------------------------------------------------------------------


def _fp16_incapable() -> hub.Device:
    return hub.Device(
        name="Google Pixel 3",
        attributes=["os:android", "chipset:qualcomm-snapdragon-845"],
    )


def test_raise_if_fp_is_unsupported_quantized_is_noop() -> None:
    """Quantized precisions never trigger the FP16 check regardless of device."""
    raise_if_fp_is_unsupported(_fp16_incapable(), Precision.w8a8)


def test_raise_if_fp_is_unsupported_rejects_fp_on_no_fp16_device() -> None:
    """FP precision on a fully-resolved fp16-incapable device raises."""
    with pytest.raises(CLIUserError, match="FP16 support"):
        raise_if_fp_is_unsupported(_fp16_incapable(), Precision.float)


# ---------------------------------------------------------------------------
# assert_success_and_get_target_models
# ---------------------------------------------------------------------------


def _mock_job(success: bool, target_model: object = None) -> hub.CompileJob:
    """A job whose target-model handle exists at submission (qai_hub 0.56+)
    regardless of whether the job ultimately succeeds.
    """
    job = MagicMock(spec=hub.CompileJob)
    job.wait.return_value.success = success
    job.get_target_model.return_value = target_model if target_model else MagicMock()
    job.url = "https://example.test/job"
    # `name` is a reserved MagicMock constructor kwarg, so it must be set after
    # construction rather than passed as `name=...` above.
    job.name = "test_job"
    return job


def test_assert_success_and_get_target_models_raises_on_failed_job() -> None:
    """A failed job must raise even though get_target_model() returns non-None.

    qai_hub 0.56 populates the target-model handle at submission time, not just
    on success, so an `is not None` check alone can no longer detect failure --
    only an explicit wait + status check can.
    """
    with pytest.raises(AssertionError, match="Job failed"):
        assert_success_and_get_target_models(_mock_job(success=False))


def test_assert_success_and_get_target_models_passes_on_success() -> None:
    job = _mock_job(success=True)
    mock_job = cast(MagicMock, job)
    result = assert_success_and_get_target_models(job)
    assert result is mock_job.get_target_model.return_value
    mock_job.wait.assert_called_once()


def test_assert_success_and_get_target_models_raises_for_one_failed_component() -> None:
    """A ComponentGroup with one failed job must raise, not silently drop it."""
    jobs = ComponentGroup({"a": _mock_job(success=True), "b": _mock_job(success=False)})
    with pytest.raises(AssertionError, match="Job failed for b"):
        assert_success_and_get_target_models(jobs)
