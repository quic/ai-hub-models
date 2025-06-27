# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest
import qai_hub as hub

from qai_hub_models.models._shared.llama3 import test
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r import MODEL_ID, Model
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.demo import (
    llama_v3_1_sea_lion_3_5_8b_r_chat_demo,
)
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.export import (
    main as export_main,
)
from qai_hub_models.utils.model_cache import CacheMode


@pytest.mark.skip("#105 move slow_cloud and slow tests to nightly.")
@pytest.mark.slow_cloud
def test_demo() -> None:
    # Run demo and verify it does not crash
    llama_v3_1_sea_lion_3_5_8b_r_chat_demo(is_test=True)


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "skip_inferencing, skip_profiling",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_cli_device_with_skips(
    tmp_path,
    skip_inferencing,
    skip_profiling,
):
    test.test_cli_device_with_skips(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        hub.Device(DEFAULT_EXPORT_DEVICE),
        skip_inferencing,
        skip_profiling,
    )


def test_cli_device_with_skips_unsupported_device(
    tmp_path,
):
    test.test_cli_device_with_skips_unsupported_device(
        export_main, Model, tmp_path, MODEL_ID
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "chipset, context_length, sequence_length",
    [
        ("qualcomm-snapdragon-8gen2", 2048, 256),
        ("qualcomm-snapdragon-x-elite", 4096, 128),
    ],
)
def test_cli_chipset_with_options(
    tmp_path,
    context_length,
    sequence_length,
    chipset,
):
    test.test_cli_chipset_with_options(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        chipset,
        context_length,
        sequence_length,
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "cache_mode, skip_download, skip_summary",
    [
        (CacheMode.ENABLE, True, True),
        (CacheMode.DISABLE, True, False),
        (CacheMode.OVERWRITE, False, False),
    ],
)
def test_cli_default_device_select_component(
    tmp_path,
    cache_mode,
    skip_download,
    skip_summary,
):
    test.test_cli_default_device_select_component(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        hub.Device(DEFAULT_EXPORT_DEVICE),
        cache_mode,
        skip_download,
        skip_summary,
    )
