# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import tempfile

import pandas as pd

from qai_hub_models.configs.numerics_yaml import (
    QAIHMModelNumerics,
    get_numerics_yaml_path,
)
from qai_hub_models.scorecard.results.yaml import ACCURACY_CSV_BASE
from qai_hub_models.scripts.create_numerics_yaml import create_numerics_struct
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.path_helpers import MODEL_IDS


def test_accuracy_yaml():
    for model_id in MODEL_IDS:
        try:
            accuracy = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
            if accuracy is None:
                continue
        except Exception as err:
            assert False, f"{model_id} numerics yaml validation failed: {str(err)}"


def test_yaml_roundtrip():
    model_id = "resnet18"
    accuracy_yaml_path = get_numerics_yaml_path(model_id)

    # Read from YAML and export back to YAML.
    # Compare both YAML dictionaries to make sure they're the same.
    original_yaml = load_yaml(accuracy_yaml_path)

    # Load Accuracy Object
    accuracy = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
    assert accuracy is not None

    # Roundtrip back to dict
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_path = os.path.join(tmp, "test.yml")
        accuracy.to_yaml(test_yaml_path)
        roundtrip_dict = load_yaml(test_yaml_path)

    assert original_yaml == roundtrip_dict


def test_accuracy_yaml_creation():
    model_id = "resnet18"
    accuracy_yaml_path = get_numerics_yaml_path(model_id)
    original_yaml = load_yaml(accuracy_yaml_path)

    # Create accuracy struct from scratch using accuracy.csv
    new_struct = create_numerics_struct(model_id, pd.read_csv(ACCURACY_CSV_BASE))
    assert new_struct is not None

    # Write to yaml and load to dict
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_path = os.path.join(tmp, "test.yml")
        new_struct.to_yaml(test_yaml_path)
        roundtrip_dict = load_yaml(test_yaml_path)

    assert original_yaml == roundtrip_dict
