# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pandas as pd

from qai_hub_models.scorecard.results.yaml import ACCURACY_CSV_BASE


def test_validate_accuracy_csv():
    df = pd.read_csv(ACCURACY_CSV_BASE)
    assert sorted(df.columns) == [
        "Device Accuracy",
        "PSNR_0",
        "PSNR_1",
        "PSNR_2",
        "PSNR_3",
        "PSNR_4",
        "PSNR_5",
        "PSNR_6",
        "PSNR_7",
        "PSNR_8",
        "PSNR_9",
        "Sim Accuracy",
        "Torch Accuracy",
        "branch",
        "chipset",
        "date",
        "model_id",
        "precision",
        "runtime",
    ]
