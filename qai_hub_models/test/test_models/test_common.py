# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest
from qai_hub import QuantizeDtype

from qai_hub_models.models.common import Precision


def test_precision_has_float():
    assert Precision.float.has_float_activations
    assert Precision.float.has_float_weights
    assert not Precision.w8a8.has_float_activations
    assert not Precision.w8a8.has_float_weights


def test_precision_eq():
    assert Precision.float == Precision.float
    assert Precision.float == Precision(None, None)
    assert Precision.float != Precision.w8a8
    assert Precision.w8a8 == Precision(QuantizeDtype.INT8, QuantizeDtype.INT8)
    assert Precision.w8a16 == Precision(QuantizeDtype.INT8, QuantizeDtype.INT16)
    assert Precision.w8a16 != Precision(QuantizeDtype.INT16, QuantizeDtype.INT8)
    assert Precision.float != 2


def test_precision_parse_serialize():
    assert str(Precision.float) == "float"
    assert str(Precision.w8a8) == "w8a8"
    assert str(Precision.w8a16) == "w8a16"

    assert Precision.from_string("float") == Precision.float

    assert Precision.from_string("w8a8") == Precision.w8a8
    assert Precision.from_string("a8w8") == Precision.w8a8

    assert Precision.from_string("w8a16") == Precision.w8a16
    assert Precision.from_string("a16w8") == Precision.w8a16

    assert Precision.from_string("w4") == Precision(QuantizeDtype.INT4, None)
    assert str(Precision.from_string("w4")) == "w4"

    assert Precision.from_string("a16") == Precision(None, QuantizeDtype.INT16)
    assert str(Precision.from_string("a16")) == "a16"

    # Invalid bit width
    with pytest.raises(ValueError):
        Precision.from_string("w8a24")
