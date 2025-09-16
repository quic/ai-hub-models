# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llama3 import test
from qai_hub_models.models._shared.llama3.model import Llama3Base
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import MainLLMInputType, cleanup
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r import MODEL_ID, FP_Model, Model
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.demo import (
    llama_v3_1_sea_lion_3_5_8b_r_chat_demo,
)
from qai_hub_models.models.llama_v3_1_sea_lion_3_5_8b_r.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PRECISION,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = 2048


# Dummy Model
class TestLlama3_SEA_LION(FP_Model):
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        llm_config.num_hidden_layers = 1
        return llm_config

    def _verify_ckpt(self):
        pass

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

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(1)


@pytest.fixture
def setup_dummy_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"dummy_{MODEL_ID}_ckpt")
    yield test.setup_test_quantization(
        Model,
        TestLlama3_SEA_LION,
        path,
        precision=DEFAULT_PRECISION,
        num_samples=1,
    )
    cleanup()


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="This test can be run on CPU only."
)
def test_cpu() -> None:
    with pytest.raises(ValueError, match=r"Please re-try with GPU machine."):
        quantize(
            quantized_model_cls=Model,
            fp_model_cls=TestLlama3_SEA_LION,
            context_length=128,
            seq_len=64,
            precision=DEFAULT_PRECISION,
            output_dir="fail_on_cpu",
            checkpoint=None,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("task", ["wikitext", "mmlu"])
def test_evaluate_dummy(
    task: str, setup_dummy_quantized_checkpoints: CheckpointSpec
) -> None:
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=TestLlama3_SEA_LION,
        num_samples=2,
        task=task,
        kwargs=dict(
            checkpoint=setup_dummy_quantized_checkpoints,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    assert isinstance(actual_metric, float) and actual_metric >= 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_demo_dummy(setup_dummy_quantized_checkpoints: CheckpointSpec) -> None:
    llama_v3_1_sea_lion_3_5_8b_r_chat_demo(
        fp_model_cls=TestLlama3_SEA_LION,
        test_checkpoint=Path(setup_dummy_quantized_checkpoints),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 8.78, 30),
        ("mmlu", 0.655, 200),
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
        ("wikitext", 7.62, 0),
        ("tiny_mmlu", 0.72, 0),
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
    llama_v3_1_sea_lion_3_5_8b_r_chat_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
