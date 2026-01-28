# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from qai_hub_models.models._shared.bert_hf.demo import bert_demo
from qai_hub_models.models.mobile_bert_uncased_google.model import (
    MODEL_ID,
    MobileBertUncasedGoogle,
)

DEMO_TEXT = "Paris is the [MASK] of France."


def main(is_test: bool = False):
    """Demo for BERT base uncased model"""
    bert_demo(MobileBertUncasedGoogle, MODEL_ID, DEMO_TEXT, is_test)


if __name__ == "__main__":
    main()
