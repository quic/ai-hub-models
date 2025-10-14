# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch

from qai_hub_models.models._shared.llama3 import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.export import export_model
from qai_hub_models.models._shared.llm.model import cleanup
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.falcon_v3_7b_instruct import (
    MODEL_ID,
    FP_Model,
    Model,
    PositionProcessor,
)
from qai_hub_models.models.falcon_v3_7b_instruct.demo import falcon_v3_7b_instruct_demo
from qai_hub_models.models.falcon_v3_7b_instruct.export import (
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    SUPPORTED_PRECISION_RUNTIMES,
)
from qai_hub_models.models.falcon_v3_7b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    MODEL_ASSET_VERSION,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    pytest_device_idfn,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.testing import allow_few_test_devices_for_llms
from qai_hub_models.utils.testing_export_eval import compile_via_export

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 7.599, 0),
        ("mmlu", 0.703, 1000),
        ("tiny_mmlu", 0.64, 0),
    ],
)
def test_evaluate_default(
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    cleanup()
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
        num_samples=num_samples,
        task=task,
        kwargs=dict(
            checkpoint="DEFAULT",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 7.453, 0),
        ("mmlu", 0.728, 1000),
        ("tiny_mmlu", 0.68, 0),
    ],
)
def test_evaluate_default_unquantized(
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    cleanup()
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
        num_samples=num_samples,
        task=task,
        kwargs=dict(
            checkpoint="DEFAULT_UNQUANTIZED",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(checkpoint: CheckpointSpec, capsys) -> None:
    cleanup()
    falcon_v3_7b_instruct_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    "precision,scorecard_path,device",
    get_compile_parameterized_pytest_config(
        MODEL_ID,
        SUPPORTED_PRECISION_RUNTIMES,
        {},
        can_use_quantize_job=False,
        only_include_genai_paths=True,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    genie_bundle_path = f"genie_bundle/{MODEL_ID}/{device.name}_{str(precision)}"
    allow_few_test_devices_for_llms(scorecard_path.runtime, device)
    compile_via_export(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint="DEFAULT",
            sequence_length=128,
            context_length=DEFAULT_CONTEXT_LENGTH,
            _skip_quantsim_creation=True,
            model_cls=Model,
            model_name=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            num_splits=NUM_SPLITS,
            num_layers_per_split=NUM_LAYERS_PER_SPLIT,
            output_dir=genie_bundle_path,
            fp_model=FP_Model.from_pretrained(
                sequence_length=128, context_length=DEFAULT_CONTEXT_LENGTH
            ),
            position_processor_cls=PositionProcessor,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(genie_bundle_path)
    assert os.path.exists(os.path.join(genie_bundle_path, "tokenizer.json"))
    assert os.path.exists(os.path.join(genie_bundle_path, "genie_config.json"))
    assert os.path.exists(
        os.path.join(genie_bundle_path, "htp_backend_ext_config.json")
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not importlib.util.find_spec("qdc_public_api_client"),
    reason="This test can be run on GPU only. Also needs QDC package to run.",
)
@pytest.mark.parametrize(
    "precision,scorecard_path,device",
    get_compile_parameterized_pytest_config(
        MODEL_ID,
        SUPPORTED_PRECISION_RUNTIMES,
        {},
        can_use_quantize_job=False,
        only_include_genai_paths=True,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.qdc
def test_qdc(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    allow_few_test_devices_for_llms(scorecard_path.runtime, device)
    genie_bundle_path = f"genie_bundle/{MODEL_ID}/{device.name}_{str(precision)}"
    if scorecard_path.runtime == TargetRuntime.ONNXRUNTIME_GENAI:
        pytest.skip("This test is only valid for Genie runtime.")
    if not os.path.exists(genie_bundle_path):
        pytest.fail("The genie bundle does not exist.")
    test.complete_genie_bundle_and_run_on_device(device, genie_bundle_path)
