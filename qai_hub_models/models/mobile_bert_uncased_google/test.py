# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from typing import cast

from transformers import PreTrainedTokenizer

from qai_hub_models.models._shared.bert_hf.app import (
    BaseBertApp,
)
from qai_hub_models.models.mobile_bert_uncased_google.demo import DEMO_TEXT
from qai_hub_models.models.mobile_bert_uncased_google.demo import main as demo_main
from qai_hub_models.models.mobile_bert_uncased_google.model import (
    MobileBertUncasedGoogle,
)
from qai_hub_models.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    model_cls = MobileBertUncasedGoogle.from_pretrained()
    tokenizer = cast(PreTrainedTokenizer, model_cls.tokenizer)
    app = BaseBertApp(model_cls, tokenizer)
    app_output = app.fill_mask(DEMO_TEXT)
    expected_text = "paris is the capital of france."
    assert app_output == expected_text


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
