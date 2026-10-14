# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf

# whisper_tiny ships per-chipset context binaries, so the filter applies.
AOT_MODEL = "whisper_tiny"
# act ships universal assets, so the filter must leave its card alone.
JIT_MODEL = "act"


def test_drop_chipsets_without_assets_drops_unbacked() -> None:
    """A claim no published asset backs must not survive."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.supported_chipsets = [*perf.supported_chipsets, "qualcomm-snapdragon-888"]
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert "qualcomm-snapdragon-888" not in perf.supported_chipsets


def test_drop_chipsets_without_assets_keeps_backed() -> None:
    """A chipset the model ships an asset for is kept even if unprofiled."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    before = set(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert "qualcomm-snapdragon-8gen1" in perf.supported_chipsets
    assert set(perf.supported_chipsets) <= before


def test_drop_chipsets_without_assets_does_not_add() -> None:
    """The filter only subtracts; it never introduces a new claim."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    before = set(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert not set(perf.supported_chipsets) - before


def test_drop_chipsets_without_assets_rewrites_devices() -> None:
    """supported_devices is derived from the kept chipsets, not left stale."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.supported_chipsets = [*perf.supported_chipsets, "qualcomm-snapdragon-888"]
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert not any("Galaxy S21" in str(d) for d in perf.supported_devices)


def test_drop_chipsets_without_assets_skips_universal_assets() -> None:
    """A universal asset runs anywhere, so its card must be untouched."""
    perf = QAIHMModelPerf.from_model(JIT_MODEL)
    before = list(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(JIT_MODEL)
    assert perf.supported_chipsets == before


def test_drop_chipsets_without_assets_is_idempotent() -> None:
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    once = list(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert perf.supported_chipsets == once
