# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qai_hub as hub
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llama3 import test
from qai_hub_models.models._shared.llama3.model import Llama3Base
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import MainLLMInputType, cleanup
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.llama_v3_8b_instruct import MODEL_ID, FP_Model, Model
from qai_hub_models.models.llama_v3_8b_instruct.demo import llama_3_chat_demo
from qai_hub_models.models.llama_v3_8b_instruct.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.llama_v3_8b_instruct.export import main as export_main
from qai_hub_models.models.llama_v3_8b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PRECISION,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.model_cache import CacheMode

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "skip_inferencing, skip_profiling, target_runtime",
    [
        (True, True, TargetRuntime.GENIE),
        (True, False, TargetRuntime.GENIE),
        (False, True, TargetRuntime.GENIE),
        (False, False, TargetRuntime.GENIE),
        (False, True, TargetRuntime.ONNXRUNTIME_GENAI),
        (False, False, TargetRuntime.ONNXRUNTIME_GENAI),
    ],
)
def test_cli_device_with_skips(
    tmp_path: Path,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
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
        target_runtime,
    )


def test_cli_device_with_skips_unsupported_device(
    tmp_path: Path,
):
    test.test_cli_device_with_skips_unsupported_device(
        export_main, Model, tmp_path, MODEL_ID
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "chipset, context_length, sequence_length, target_runtime",
    [
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.GENIE),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.GENIE),
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.ONNXRUNTIME_GENAI),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.ONNXRUNTIME_GENAI),
    ],
)
def test_cli_chipset_with_options(
    tmp_path: Path,
    context_length: int,
    sequence_length: int,
    chipset: str,
    target_runtime: TargetRuntime,
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
        target_runtime,
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "cache_mode, skip_download, skip_summary, target_runtime",
    [
        (CacheMode.ENABLE, True, True, TargetRuntime.GENIE),
        (CacheMode.DISABLE, True, False, TargetRuntime.GENIE),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.GENIE),
        (CacheMode.ENABLE, True, True, TargetRuntime.ONNXRUNTIME_GENAI),
        (CacheMode.DISABLE, True, False, TargetRuntime.ONNXRUNTIME_GENAI),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.ONNXRUNTIME_GENAI),
    ],
)
def test_cli_default_device_select_component(
    tmp_path: Path,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
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
        target_runtime,
    )


# Dummy Model
class TestLlama3(FP_Model):
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        llm_config.num_hidden_layers = 1
        return llm_config

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        main_input_type: MainLLMInputType = MainLLMInputType.input_ids,
    ) -> FP_Model:
        return cls(
            checkpoint=HF_REPO_NAME,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device="cpu",
            load_pretrained=False,
            main_input_type=main_input_type,
        )

    def _verify_ckpt(self):
        pass

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(1)


@pytest.fixture(scope="session")
def setup_dummy_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"{MODEL_ID}_dummy_ckpt")
    yield test.setup_test_quantization(
        Model,
        TestLlama3,
        path,
        precision=DEFAULT_PRECISION,
        num_samples=1,
    )
    cleanup()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("task", ["wikitext", "mmlu"])
def test_evaluate_dummy(
    task: str, setup_dummy_quantized_checkpoints: CheckpointSpec
) -> None:
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=TestLlama3,
        num_samples=2,
        task=task,
        kwargs=dict(
            checkpoint=Path(setup_dummy_quantized_checkpoints),
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    assert isinstance(actual_metric, float) and actual_metric >= 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_demo_dummy(setup_dummy_quantized_checkpoints: CheckpointSpec) -> None:
    llama_3_chat_demo(
        fp_model_cls=TestLlama3,
        test_checkpoint=Path(setup_dummy_quantized_checkpoints),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 8.736, 0),
        ("mmlu", 0.618, 1000),
        ("tiny_mmlu", 0.53, 0),
    ],
)
def test_evaluate_default(
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 7.76, 0),
        ("mmlu", 0.674, 1000),
        ("tiny_mmlu", 0.61, 0),
    ],
)
def test_evaluate_default_unquantized(
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(checkpoint: CheckpointSpec, capsys) -> None:
    llama_3_chat_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
