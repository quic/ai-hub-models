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
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.llama_v3_taide_8b_chat import MODEL_ID, FP_Model, Model
from qai_hub_models.models.llama_v3_taide_8b_chat.demo import (
    llama_v3_taide_8b_chat_demo,
)
from qai_hub_models.models.llama_v3_taide_8b_chat.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.llama_v3_taide_8b_chat.export import main as export_main
from qai_hub_models.models.llama_v3_taide_8b_chat.model import (
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
class TestLlama3_TAIDE(FP_Model):
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
    ) -> FP_Model:
        return cls(
            checkpoint=HF_REPO_NAME,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device="cpu",
            load_pretrained=False,
        )

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(1)


@pytest.fixture(scope="session")
def setup_dummy_quantized_checkpoints(tmpdir_factory):
    path = tmpdir_factory.mktemp(f"dummy_{MODEL_ID}_ckpt")
    return test.setup_test_quantization(
        Model,
        TestLlama3_TAIDE,
        path,
        precision=DEFAULT_PRECISION,
        num_samples=1,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("task", ["wikitext-ppl", "mmlu"])
def test_evaluate_dummy(
    task: str, setup_dummy_quantized_checkpoints: CheckpointSpec
) -> None:
    model, is_quantized, host_device = create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=TestLlama3_TAIDE,
        kwargs=dict(
            _skip_quantsim_creation=False,
            checkpoint=setup_dummy_quantized_checkpoints,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            fp_model=None,
        ),
    )
    actual_metric, _ = evaluate(
        fp_model_cls=TestLlama3_TAIDE,
        num_samples=2,
        task=task,
        kwargs=dict(
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
        model=model,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    assert isinstance(actual_metric, float) and actual_metric >= 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_model_dummy(setup_dummy_quantized_checkpoints: CheckpointSpec) -> None:
    llama_v3_taide_8b_chat_demo(
        fp_model_cls=TestLlama3_TAIDE,
        test_checkpoint=Path(setup_dummy_quantized_checkpoints),
    )


@pytest.fixture(scope="session")
def setup_create_quantsim_default():
    return create_quantsim(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
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
        fp_model_cls=FP_Model,
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
        ("mmlu", 0.503, 1000),
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
        fp_model_cls=FP_Model,
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
        ("mmlu", 0.57, 1000),
        ("tiny-mmlu", 0.61, 0),
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
        fp_model_cls=FP_Model,
        is_quantized=is_quantized,
        host_device=host_device,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)
