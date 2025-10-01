# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import qai_hub as hub
import torch
from transformers import AutoConfig, PretrainedConfig

from qai_hub_models.models._shared.llama3 import test
from qai_hub_models.models._shared.llama3.model import Llama3Base
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import MainLLMInputType, cleanup
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.llama_v3_2_1b_instruct import MODEL_ID, FP_Model, Model
from qai_hub_models.models.llama_v3_2_1b_instruct.demo import llama_3_2_1b_chat_demo
from qai_hub_models.models.llama_v3_2_1b_instruct.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.llama_v3_2_1b_instruct.export import main as export_main
from qai_hub_models.models.llama_v3_2_1b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PRECISION,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
)
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.llm_helpers import create_genie_config
from qai_hub_models.utils.model_cache import CacheMode

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.parametrize(
    "model_name, rope_dim, kv_dim, has_pos_enc, eos_token",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", 32, 64, True, [128001, 128008, 128009]),
        ("meta-llama/Llama-3.2-3B-Instruct", 64, 128, True, [128001, 128008, 128009]),
        ("meta-llama/Meta-Llama-3-8B-Instruct", 64, 128, False, 128009),
    ],
)
def test_create_genie_config(
    model_name: str, rope_dim: int, kv_dim: int, has_pos_enc: bool, eos_token: list[int]
):
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(model_name)
    model_list = ["model1.bin", "model2.bin"]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 128256,
                "bos-token": 128000,
                "eos-token": eos_token,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": False,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": kv_dim,
                        "allow-async-init": False,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": [
                            "model1.bin",
                            "model2.bin",
                        ],
                    },
                },
            },
        }
    }
    if has_pos_enc:
        expected_config["dialog"]["engine"]["model"]["positional-encodings"] = {
            "type": "rope",
            "rope-dim": rope_dim,
            "rope-theta": 500000,
            "rope-scaling": {
                "rope-type": "llama3",
                "factor": 8.0,
                "low-freq-factor": 1.0,
                "high-freq-factor": 4.0,
                "original-max-position-embeddings": 8192,
            },
        }

    assert expected_config == actual_config


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "skip_inferencing, skip_profiling, target_runtime",
    [
        (True, True, TargetRuntime.GENIE),
        (True, False, TargetRuntime.GENIE),
        (False, True, TargetRuntime.GENIE),
        (False, False, TargetRuntime.GENIE),
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
        precision=Precision.w4,
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "cache_mode, skip_download, skip_summary, target_runtime",
    [
        (CacheMode.ENABLE, True, True, TargetRuntime.GENIE),
        (CacheMode.DISABLE, True, False, TargetRuntime.GENIE),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.GENIE),
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


# Dummy model
class TestLlama3_2(FP_Model):
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        llm_config.num_hidden_layers = 1
        return llm_config

    def _verify_ckpt(self):
        pass

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(1)

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
            load_pretrained=False,
            main_input_type=main_input_type,
        )


@pytest.fixture(scope="session")
def setup_dummy_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"dummy_{MODEL_ID}_ckpt")
    yield test.setup_test_quantization(
        Model,
        TestLlama3_2,
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
            fp_model_cls=TestLlama3_2,
            context_length=128,
            seq_len=64,
            precision=DEFAULT_PRECISION,
            output_dir="fail_on_cpu",
            checkpoint=None,
        )


@pytest.mark.parametrize("task", ["wikitext", "mmlu"])
def test_evaluate_dummy(task: str, setup_dummy_quantized_checkpoints) -> None:
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=TestLlama3_2,
        num_samples=2,
        task=task,
        kwargs=dict(
            checkpoint=setup_dummy_quantized_checkpoints,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    assert isinstance(actual_metric, float) and actual_metric >= 0.0


def test_demo_dummy(setup_dummy_quantized_checkpoints: CheckpointSpec) -> None:
    llama_3_2_1b_chat_demo(
        fp_model_cls=TestLlama3_2,
        test_checkpoint=setup_dummy_quantized_checkpoints,
    )


# Full model tests
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint):
    Model.from_pretrained(fp_model=FP_Model.from_pretrained())


@pytest.fixture(scope="session")
def setup_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"{MODEL_ID}_ai_nexuz_ckpt")
    yield test.setup_test_quantization(
        Model,
        FP_Model,
        path,
        precision=DEFAULT_PRECISION,
        checkpoint="ai-nexuz/llama-3.2-1b-instruct-fine-tuned",
    )
    cleanup()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 16.78, 0),
        ("mmlu", 0.399, 1000),
        ("tiny_mmlu", 0.43, 0),
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
        ("wikitext", 12.18, 0),
        ("mmlu", 0.482, 1000),
        ("tiny_mmlu", 0.41, 0),
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
            _skip_quantsim_creation=False,
            checkpoint="DEFAULT_UNQUANTIZED",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext", 17.29, 0),
        ("mmlu", 0.397, 1000),
        ("tiny_mmlu", 0.34, 0),
    ],
)
def test_evaluate_quantsim_default_w4a16(
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
            _skip_quantsim_creation=False,
            checkpoint="DEFAULT_W4A16",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(checkpoint: CheckpointSpec, capsys) -> None:
    llama_3_2_1b_chat_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_demo_quantized_checkpoint(
    setup_quantized_checkpoints: CheckpointSpec, capsys
) -> None:
    llama_3_2_1b_chat_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=setup_quantized_checkpoints,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
