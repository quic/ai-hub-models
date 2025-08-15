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
from qai_hub_models.models._shared.llm.evaluate import create_quantsim, evaluate
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.llama_v3_2_3b_instruct import MODEL_ID, Model
from qai_hub_models.models.llama_v3_2_3b_instruct.demo import llama_3_2_3b_chat_demo
from qai_hub_models.models.llama_v3_2_3b_instruct.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.llama_v3_2_3b_instruct.export import main as export_main
from qai_hub_models.models.llama_v3_2_3b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PRECISION,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
    Llama3_2_3B,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.model_cache import CacheMode

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
@pytest.mark.parametrize(
    "skip_inferencing, skip_profiling, target_runtime",
    [
        (True, True, TargetRuntime.QNN_CONTEXT_BINARY),
        (True, False, TargetRuntime.QNN_CONTEXT_BINARY),
        (False, True, TargetRuntime.QNN_CONTEXT_BINARY),
        (False, False, TargetRuntime.QNN_CONTEXT_BINARY),
        (False, True, TargetRuntime.PRECOMPILED_QNN_ONNX),
        (False, False, TargetRuntime.PRECOMPILED_QNN_ONNX),
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
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.QNN_CONTEXT_BINARY),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.QNN_CONTEXT_BINARY),
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.PRECOMPILED_QNN_ONNX),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.PRECOMPILED_QNN_ONNX),
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
        (CacheMode.ENABLE, True, True, TargetRuntime.QNN_CONTEXT_BINARY),
        (CacheMode.DISABLE, True, False, TargetRuntime.QNN_CONTEXT_BINARY),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.QNN_CONTEXT_BINARY),
        (CacheMode.ENABLE, True, True, TargetRuntime.PRECOMPILED_QNN_ONNX),
        (CacheMode.DISABLE, True, False, TargetRuntime.PRECOMPILED_QNN_ONNX),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.PRECOMPILED_QNN_ONNX),
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
class TestLlama3_2(Llama3_2_3B):
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
    ) -> Llama3_2_3B:
        return cls(
            checkpoint=HF_REPO_NAME,
            sequence_length=sequence_length,
            context_length=context_length,
            load_pretrained=False,
        )


@pytest.fixture(scope="session")
def setup_dummy_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"dummy_{MODEL_ID}_ckpt")
    return test.setup_test_quantization(
        Model,
        TestLlama3_2,
        path,
        precision=DEFAULT_PRECISION,
        num_samples=1,
    )


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


@pytest.mark.parametrize("task", ["wikitext-ppl", "mmlu"])
def test_dummy_model_evaluate(
    task: str, setup_dummy_quantized_checkpoints: CheckpointSpec
) -> None:
    model, is_quantized, host_device = create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=TestLlama3_2,
        kwargs=dict(
            _skip_quantsim_creation=False,
            checkpoint=setup_dummy_quantized_checkpoints,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )
    actual_metric, _ = evaluate(
        fp_model_cls=TestLlama3_2,
        num_samples=2,
        task=task,
        kwargs=dict(
            context_length=4096,
        ),
        model=model,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    assert isinstance(actual_metric, float) and actual_metric >= 0.0


def test_dummy_model_demo(setup_dummy_quantized_checkpoints: CheckpointSpec) -> None:
    llama_3_2_3b_chat_demo(
        fp_model_cls=TestLlama3_2,
        test_checkpoint=setup_dummy_quantized_checkpoints,
    )


# Full model tests
@pytest.fixture(scope="session")
def setup_quantized_checkpoints(tmpdir_factory) -> str:
    path = tmpdir_factory.mktemp(f"{MODEL_ID}_korean")
    return test.setup_test_quantization(
        Model,
        Llama3_2_3B,
        path,
        precision=DEFAULT_PRECISION,
        checkpoint="Bllossom/llama-3.2-Korean-Bllossom-3B",
    )


@pytest.fixture(scope="session")
def setup_create_quantsim_default():
    return create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=Llama3_2_3B,
        kwargs=dict(
            _skip_quantsim_creation=False,
            checkpoint="DEFAULT",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )


@pytest.fixture(scope="session")
def setup_create_default_unquantized():
    return create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=Llama3_2_3B,
        kwargs=dict(
            _skip_quantsim_creation=False,
            checkpoint="DEFAULT_UNQUANTIZED",
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext-ppl", 11.997, 0),
        ("mmlu", 0.566, 1000),
        ("tiny-mmlu", 0.53, 0),
    ],
)
def test_evaluate_default(
    setup_create_quantsim_default,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    model, is_quantized, host_device = setup_create_quantsim_default
    actual_metric, _ = evaluate(
        num_samples=num_samples,
        task=task,
        model=model,
        kwargs=dict(
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
        fp_model_cls=Llama3_2_3B,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    "task,expected_metric,num_samples",
    [
        ("wikitext-ppl", 10.165, 0),
        ("mmlu", 0.613, 1000),
        ("tiny-mmlu", 0.64, 0),
    ],
)
def test_evaluate_default_unquantized(
    setup_create_default_unquantized,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    model, is_quantized, host_device = setup_create_default_unquantized
    actual_metric, _ = evaluate(
        num_samples=num_samples,
        task=task,
        model=model,
        kwargs=dict(
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
        fp_model_cls=Llama3_2_3B,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_evaluate_quantized_checkpoint(
    setup_quantized_checkpoints: str,
) -> None:
    model, is_quantized, host_device = create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=Llama3_2_3B,
        kwargs=dict(
            _skip_quantsim_creation=False,
            checkpoint=setup_quantized_checkpoints,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )
    actual_metric, _ = evaluate(
        fp_model_cls=Llama3_2_3B,
        task="mmmlu-ko",
        num_samples=100,
        kwargs=dict(
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
        model=model,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    np.testing.assert_allclose(actual_metric, 0.3, rtol=1e-02, atol=1e-02)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_demo_quantized_checkpoint(setup_quantized_checkpoints: str, capsys) -> None:
    llama_3_2_3b_chat_demo(
        fp_model_cls=Llama3_2_3B,
        default_prompt="What is the capital of France?",
        test_checkpoint=Path(setup_quantized_checkpoints),
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
