# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from dataclasses import dataclass, fields
from typing import Optional

from qai_hub_models.utils.base_config import BaseQAIHMConfig, ParseableQAIHMEnum


def test_parse_primitives() -> None:
    assert BaseQAIHMConfig.parse_field_from_type(int, 1, "") == 1
    assert BaseQAIHMConfig.parse_field_from_type(int, "1", "") == 1

    assert BaseQAIHMConfig.parse_field_from_type(float, 1.2, "") == 1.2
    assert BaseQAIHMConfig.parse_field_from_type(float, "1.2", "") == 1.2

    assert BaseQAIHMConfig.parse_field_from_type(bool, True, "")
    assert not BaseQAIHMConfig.parse_field_from_type(bool, False, "")
    assert BaseQAIHMConfig.parse_field_from_type(bool, "True", "")
    assert not BaseQAIHMConfig.parse_field_from_type(bool, "False", "")
    assert BaseQAIHMConfig.parse_field_from_type(bool, "true", "")
    assert not BaseQAIHMConfig.parse_field_from_type(bool, "false", "")

    assert BaseQAIHMConfig.parse_field_from_type(str, "asdf", "") == "asdf"
    assert BaseQAIHMConfig.parse_field_from_type(str, "1.2", "") == "1.2"
    assert BaseQAIHMConfig.parse_field_from_type(str, 1.2, "") == "1.2"


def test_parse_enum() -> None:
    class TestEnum(ParseableQAIHMEnum):
        my_apple = 1
        super_pear = 2

        @staticmethod
        def from_string(string: str) -> "TestEnum":
            return TestEnum[string.replace(" ", "_").lower()]

    assert BaseQAIHMConfig.parse_field_from_type(TestEnum, 1, "") == TestEnum.my_apple
    assert (
        BaseQAIHMConfig.parse_field_from_type(TestEnum, "my_apple", "")
        == TestEnum.my_apple
    )
    assert (
        BaseQAIHMConfig.parse_field_from_type(TestEnum, "Super Pear", "")
        == TestEnum.super_pear
    )  # Calls from_string on enum


def test_parse_optional() -> None:
    assert BaseQAIHMConfig.parse_field_from_type(Optional[int], None, "") is None
    assert BaseQAIHMConfig.parse_field_from_type(Optional[int], 1, "") == 1
    assert BaseQAIHMConfig.parse_field_from_type(Optional[int], "2", "") == 2


def test_parse_containers() -> None:
    assert BaseQAIHMConfig.parse_field_from_type(list[int], ["1", 2, 3], "") == [
        1,
        2,
        3,
    ]
    assert BaseQAIHMConfig.parse_field_from_type(
        tuple[int, float, bool], ("1", 2.0, "false"), ""
    ) == (1, 2.0, False)
    assert BaseQAIHMConfig.parse_field_from_type(
        dict[str, float], {"asdf": 1.0, "ghjk": "1.0"}, ""
    ) == {"asdf": 1.0, "ghjk": 1.0}
    assert BaseQAIHMConfig.parse_field_from_type(
        dict[str, list[tuple[int, bool]]], {"asdf": [("1", False), (2, "True")]}, ""
    ) == {"asdf": [(1, False), (2, True)]}


def test_parse_string_type() -> None:
    assert BaseQAIHMConfig.parse_field_from_type("Optional[int]", None, "") is None
    assert BaseQAIHMConfig.parse_field_from_type("int", 1, "") == 1
    assert BaseQAIHMConfig.parse_field_from_type(
        "dict[str, str]", {"hello": "world"}, ""
    ) == {"hello": "world"}


def test_parse_config() -> None:
    @dataclass
    class TestConfig(BaseQAIHMConfig):
        x: str
        z: int

    x_field, z_field = fields(TestConfig)
    assert TestConfig.parse_field(x_field, "word") == "word"
    assert TestConfig.parse_field(z_field, "1") == 1
    assert TestConfig.parse_field("z", "1") == 1
    assert TestConfig.from_dict({"x": "hello", "z": "456"}) == TestConfig("hello", 456)


def test_parse_nested_config() -> None:
    @dataclass
    class TestConfigNested(BaseQAIHMConfig):
        z: bool

    @dataclass
    class TestConfigBase(BaseQAIHMConfig):
        x: str
        y: Optional[TestConfigNested] = None

    _, y_field = fields(TestConfigBase)
    assert TestConfigBase.parse_field(y_field, {"z": True}) == TestConfigNested(True)
    assert TestConfigBase.parse_field("y", {"z": "false"}) == TestConfigNested(False)
    assert TestConfigBase.from_dict(
        {"x": "hello", "y": {"z": "false"}}
    ) == TestConfigBase("hello", TestConfigNested(False))
    assert TestConfigBase.from_dict({"x": "hello"}) == TestConfigBase("hello")
