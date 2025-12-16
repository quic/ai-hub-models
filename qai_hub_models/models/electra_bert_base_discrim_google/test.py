# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

from transformers import PreTrainedTokenizer

from qai_hub_models.models.electra_bert_base_discrim_google.app import (
    ElectraBertApp,
)
from qai_hub_models.models.electra_bert_base_discrim_google.demo import INPUT_TEXT
from qai_hub_models.models.electra_bert_base_discrim_google.demo import (
    main as demo_main,
)
from qai_hub_models.models.electra_bert_base_discrim_google.model import (
    ElectraBertBaseDiscrimGoogle,
)
from qai_hub_models.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    model_cls = ElectraBertBaseDiscrimGoogle.from_pretrained()
    tokenizer = cast(PreTrainedTokenizer, model_cls.tokenizer)
    app = ElectraBertApp(model_cls, tokenizer)
    app_output = app.detect_replacements(INPUT_TEXT)
    expected_text = "fake"
    assert app_output == expected_text


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
