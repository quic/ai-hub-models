# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from enum import Enum, unique
from pathlib import Path

from qai_hub_models.utils.envvar_bases import (
    QAIHMBoolEnvvar,
    QAIHMDateFormatEnvvar,
    QAIHMPathEnvvar,
    QAIHMStringEnvvar,
    QAIHMStringListEnvvar,
    QAIHMStrSetWithEnumEnvvar,
)


class DefaultTrueEnvvar(QAIHMBoolEnvvar):
    VARNAME = "TEST_BOOL_TRUE"
    CLI_ARGNAMES = ["--set-false", "-sf"]
    CLI_HELP_MESSAGE = ""

    @classmethod
    def default(cls):
        return True


class DefaultFalseEnvvar(QAIHMBoolEnvvar):
    VARNAME = "TEST_BOOL_FALSE"
    CLI_ARGNAMES = ["--set-true", "-st"]
    CLI_HELP_MESSAGE = ""

    @classmethod
    def default(cls):
        return False


def test_bool_envvar(monkeypatch):
    # Test value serialization
    assert DefaultTrueEnvvar.serialize(True) == "1"
    assert DefaultTrueEnvvar.serialize(False) == "0"
    assert DefaultTrueEnvvar.parse(DefaultTrueEnvvar.serialize(True))
    assert not DefaultTrueEnvvar.parse(DefaultTrueEnvvar.serialize(False))

    # Verify behavior when there is no envvar set.
    DefaultTrueEnvvar.patchenv(monkeypatch, None)
    assert DefaultTrueEnvvar.get()
    assert not DefaultTrueEnvvar.get(default=False)
    DefaultFalseEnvvar.patchenv(monkeypatch, None)
    assert not DefaultFalseEnvvar.get()
    assert DefaultFalseEnvvar.get(default=True)

    # Verify behavior when there is an envvar set.
    for value in QAIHMBoolEnvvar.TRUTHY_BOOLEAN_VALUES:
        # Parsing every value should return true.
        assert DefaultTrueEnvvar.parse(value)
        assert DefaultFalseEnvvar.parse(value)

        # Set envvar to the string value.
        DefaultTrueEnvvar.patchenv(monkeypatch, value)
        DefaultFalseEnvvar.patchenv(monkeypatch, value)

        # Classes should read the envvar and always return true.
        assert DefaultTrueEnvvar.get()
        assert DefaultFalseEnvvar.get()

        # If the user passes their own default, the envvar still takes precidence.
        assert DefaultTrueEnvvar.get(False)
        assert DefaultFalseEnvvar.get(False)

    for value in {"0", "false", "", "abc"}:
        # Parsing every value should return false.
        assert not DefaultTrueEnvvar.parse(value)
        assert not DefaultFalseEnvvar.parse(value)

        # Set envvar to the string value.
        DefaultTrueEnvvar.patchenv(monkeypatch, value)
        DefaultFalseEnvvar.patchenv(monkeypatch, value)

        # Classes should read the envvar and always return false.
        assert not DefaultTrueEnvvar.get()
        assert not DefaultFalseEnvvar.get()

        # If the user passes their own default, the envvar still takes precidence.
        assert not DefaultTrueEnvvar.get(True)
        assert not DefaultFalseEnvvar.get(True)

    # Verify parser behavior when no envvars are set.
    parser = ArgumentParser()
    DefaultTrueEnvvar.patchenv(monkeypatch, None)
    DefaultFalseEnvvar.patchenv(monkeypatch, None)
    DefaultTrueEnvvar.add_arg(parser)
    DefaultFalseEnvvar.add_arg(parser)
    assert not parser.parse_args(["--set-false"]).set_false
    assert not parser.parse_args(["-sf"]).set_false
    assert parser.parse_args([]).set_false
    assert parser.parse_args([]).set_false
    assert parser.parse_args(["--set-true"]).set_true
    assert parser.parse_args(["-st"]).set_true
    assert not parser.parse_args([]).set_true
    assert not parser.parse_args([]).set_true

    # Verify same behavior when the environment set default matches the code default.
    parser = ArgumentParser()
    DefaultTrueEnvvar.patchenv(monkeypatch, DefaultTrueEnvvar.serialize(True))
    DefaultFalseEnvvar.patchenv(monkeypatch, DefaultTrueEnvvar.serialize(False))
    DefaultTrueEnvvar.add_arg(parser)
    DefaultFalseEnvvar.add_arg(parser)
    assert not parser.parse_args(["--set-false"]).set_false
    assert not parser.parse_args(["-sf"]).set_false
    assert parser.parse_args([]).set_false
    assert parser.parse_args([]).set_false
    assert parser.parse_args(["--set-true"]).set_true
    assert parser.parse_args(["-st"]).set_true
    assert not parser.parse_args([]).set_true
    assert not parser.parse_args([]).set_true

    # Now swap the environment set default.
    # The parser should add `-no-` at the beginning of the flag to flip its meaning.
    parser = ArgumentParser()
    DefaultTrueEnvvar.patchenv(monkeypatch, False)
    DefaultFalseEnvvar.patchenv(monkeypatch, True)
    DefaultTrueEnvvar.add_arg(parser)
    DefaultFalseEnvvar.add_arg(parser)
    assert parser.parse_args(["--no-set-false"]).set_false
    assert parser.parse_args(["-no-sf"]).set_false
    assert not parser.parse_args([]).set_false
    assert not parser.parse_args([]).set_false
    assert not parser.parse_args(["--no-set-true"]).set_true
    assert not parser.parse_args(["-no-st"]).set_true
    assert parser.parse_args([]).set_true
    assert parser.parse_args([]).set_true


class DateFormatEnvvar(QAIHMDateFormatEnvvar):
    class FormatEnvvar(QAIHMDateFormatEnvvar.FormatEnvvar):
        VARNAME = "TEST_DATE_FORMAT"

        @classmethod
        def default(cls):
            return "%Y-%m-%dT%H:%M:%SZ"

    class DateEnvvar(QAIHMDateFormatEnvvar.DateEnvvar):
        VARNAME = "TEST_DATE"
        _DEFAULT_OBJ = datetime(2025, 1, 1)

        @classmethod
        def default(cls):
            return cls._DEFAULT_OBJ.strftime(
                DateFormatEnvvar.DATE_FORMAT_ENVVAR.default()
            )

    DATE_ENVVAR = DateEnvvar
    DATE_FORMAT_ENVVAR = FormatEnvvar


def test_date_format_envvar(monkeypatch):
    # fmt: off
    # Test dates
    date = datetime(2028, 10, 12)
    date2 = datetime(2029, 5, 15)
    alternative_format = "%Y-%m-%dT%H:%M"
    serialized_date = date.strftime(DateFormatEnvvar.FormatEnvvar.default())
    serialized_date_with_alt_format = date.strftime(alternative_format)
    serialized_date2 = date.strftime(DateFormatEnvvar.FormatEnvvar.default())
    serialized_date2_with_alt_format = date2.strftime(alternative_format)

    # Verify Defaults
    DateFormatEnvvar.patchenv(monkeypatch, None, None)
    assert DateFormatEnvvar.get() == DateFormatEnvvar.DateEnvvar._DEFAULT_OBJ
    assert DateFormatEnvvar.get(date2) == date2

    # Serialize & parse
    assert DateFormatEnvvar.serialize(date, DateFormatEnvvar.FormatEnvvar.default()) == serialized_date
    assert DateFormatEnvvar.serialize(date, alternative_format) == serialized_date_with_alt_format
    assert DateFormatEnvvar.serialize(date2, alternative_format) == serialized_date2_with_alt_format
    assert DateFormatEnvvar.parse(serialized_date, DateFormatEnvvar.FormatEnvvar.default()) == date
    assert DateFormatEnvvar.parse(serialized_date2_with_alt_format, alternative_format) == date2

    # Override date; test .get() and argparse
    DateFormatEnvvar.patchenv(monkeypatch, serialized_date, None)
    assert DateFormatEnvvar.get() == date
    assert DateFormatEnvvar.serialize(date, DateFormatEnvvar.FormatEnvvar.default()) == serialized_date
    parser = ArgumentParser()
    DateFormatEnvvar.add_arg_group(parser)
    assert parser.parse_args([]).date == serialized_date
    assert parser.parse_args([]).date_format == DateFormatEnvvar.FormatEnvvar.default()
    assert parser.parse_args(["--date", serialized_date2]).date == serialized_date2
    assert parser.parse_args(["--date-format", alternative_format]).date_format == alternative_format

    # Override format; test .get() and argparse
    DateFormatEnvvar.patchenv(monkeypatch, serialized_date2_with_alt_format, alternative_format)
    assert DateFormatEnvvar.get() == date2
    parser = ArgumentParser()
    DateFormatEnvvar.add_arg_group(parser)
    assert parser.parse_args([]).date == serialized_date2_with_alt_format
    assert parser.parse_args([]).date_format == alternative_format
    assert parser.parse_args(["--date", serialized_date]).date == serialized_date
    assert parser.parse_args(["--date-format", DateFormatEnvvar.FormatEnvvar.default()]).date_format == DateFormatEnvvar.FormatEnvvar.default()
    # fmt: on


class ListEnvvar(QAIHMStringListEnvvar):
    VARNAME = "TEST_LIST"
    CLI_ARGNAMES = ["--list"]
    CLI_HELP_MESSAGE = ""

    @classmethod
    def default(cls):
        return ["1", "2", "3"]


def test_list_envvar(monkeypatch):
    list1 = ["a", "b", "c"]
    list2 = ["1", "2", "3", "3"]
    list3 = ["1"]

    ListEnvvar.patchenv(monkeypatch, None)
    assert ListEnvvar.get() == ListEnvvar.default()
    assert ListEnvvar.get(list1) == list1
    assert ListEnvvar.parse(ListEnvvar.serialize(list1)) == list1
    # Argparser with standard default
    parser = ArgumentParser()
    ListEnvvar.add_arg(parser)
    assert parser.parse_args([]).list == ListEnvvar.default()
    assert parser.parse_args(["--list", "1, 2, 3"]).list == ListEnvvar.default()
    assert parser.parse_args(["--list", "1,2, 3,   3   "]).list == list2
    # Passed-in default applies because the envvar is unset.
    parser = ArgumentParser()
    ListEnvvar.add_arg(parser, default=list2)
    assert parser.parse_args([]).list == list2

    ListEnvvar.patchenv(monkeypatch, "1,2, 3,   3   ")
    assert ListEnvvar.get() == list2
    assert ListEnvvar.parse(ListEnvvar.serialize(list2)) == list2
    parser = ArgumentParser()
    ListEnvvar.add_arg(parser)
    assert parser.parse_args([]).list == list2
    assert parser.parse_args(["--list", "1, 2, 3"]).list == ListEnvvar.default()
    assert parser.parse_args(["--list", "1,2, 3,   3   "]).list == list2

    ListEnvvar.patchenv(monkeypatch, "1")
    assert ListEnvvar.get() == list3
    assert ListEnvvar.parse(ListEnvvar.serialize(list3)) == list3
    # Argparser with standard default
    parser = ArgumentParser()
    ListEnvvar.add_arg(parser)
    assert parser.parse_args([]).list == list3
    assert parser.parse_args(["--list", "1"]).list == list3
    assert parser.parse_args(["--list", "1,2, 3,   3   "]).list == list2
    parser = ArgumentParser()
    # Passed-in default does not apply since the envvar is set.
    ListEnvvar.add_arg(parser, default=list2)
    assert parser.parse_args([]).list == list3


class PathEnvvar(QAIHMPathEnvvar):
    VARNAME = "TEST_PATH"
    CLI_ARGNAMES = ["--path"]
    CLI_HELP_MESSAGE = ""

    @classmethod
    def default(cls):
        return Path("/asdf")


def test_path_envvar(monkeypatch):
    path1 = Path("/hello")
    path2 = Path("/world")
    path3 = Path("hello_world")

    assert PathEnvvar.parse(PathEnvvar.serialize(path1)) == path1
    assert PathEnvvar.parse(PathEnvvar.serialize(path2)) == path2
    assert PathEnvvar.parse(PathEnvvar.serialize(path3)) == path3

    PathEnvvar.patchenv(monkeypatch, None)
    assert PathEnvvar.get() == PathEnvvar.default()
    assert PathEnvvar.get(path1) == path1

    # Argparser with standard default
    parser = ArgumentParser()
    PathEnvvar.add_arg(parser)
    assert parser.parse_args([]).path == PathEnvvar.default()
    assert parser.parse_args(["--path", "/asdf"]).path == PathEnvvar.default()
    assert parser.parse_args(["--path", "/hello"]).path == path1

    # Passed-in default applies because the envvar is unset.
    parser = ArgumentParser()
    PathEnvvar.add_arg(parser, default=path1)
    assert parser.parse_args([]).path == path1

    PathEnvvar.patchenv(monkeypatch, path1)
    assert PathEnvvar.get() == path1
    assert PathEnvvar.get(path2) == path1

    # Argparser with standard default
    parser = ArgumentParser()
    PathEnvvar.add_arg(parser)
    assert parser.parse_args([]).path == path1
    assert parser.parse_args(["--path", "/asdf"]).path == PathEnvvar.default()
    assert parser.parse_args(["--path", "/hello"]).path == path1

    # Passed-in default does not apply since the envvar is set.
    parser = ArgumentParser()
    PathEnvvar.add_arg(parser, default=path2)
    assert parser.parse_args([]).path == path1


class StringEnvvar(QAIHMStringEnvvar):
    VARNAME = "TEST_STRING"
    CLI_ARGNAMES = ["--str"]
    CLI_HELP_MESSAGE = ""

    @classmethod
    def default(cls):
        return "/asdf"


def test_string_envvar(monkeypatch):
    str1 = "/hello"
    str2 = "/world"
    str3 = "hello_world"

    assert StringEnvvar.parse(StringEnvvar.serialize(str1)) == str1
    assert StringEnvvar.parse(StringEnvvar.serialize(str2)) == str2
    assert StringEnvvar.parse(StringEnvvar.serialize(str3)) == str3

    StringEnvvar.patchenv(monkeypatch, None)
    assert StringEnvvar.get() == StringEnvvar.default()
    assert StringEnvvar.get(str1) == str1

    # Argparser with standard default
    parser = ArgumentParser()
    StringEnvvar.add_arg(parser)
    assert parser.parse_args([]).str == StringEnvvar.default()
    assert parser.parse_args(["--str", "/asdf"]).str == StringEnvvar.default()
    assert parser.parse_args(["--str", "/hello"]).str == str1

    # Passed-in default applies because the envvar is unset.
    parser = ArgumentParser()
    StringEnvvar.add_arg(parser, default=str1)
    assert parser.parse_args([]).str == str1

    StringEnvvar.patchenv(monkeypatch, str1)
    assert StringEnvvar.get() == str1
    assert StringEnvvar.get(str2) == str1

    # Argparser with standard default
    parser = ArgumentParser()
    StringEnvvar.add_arg(parser)
    assert parser.parse_args([]).str == str1
    assert parser.parse_args(["--str", "/asdf"]).str == StringEnvvar.default()
    assert parser.parse_args(["--str", "/hello"]).str == str1

    # Passed-in default does not apply since the envvar is set.
    parser = ArgumentParser()
    StringEnvvar.add_arg(parser, default=str2)
    assert parser.parse_args([]).str == str1


@unique
class SpecialEnvvarTestSetting(Enum):
    x = "1"
    y = "2"
    z = "3"


class TestEnumEnvvarSet(QAIHMStrSetWithEnumEnvvar[SpecialEnvvarTestSetting]):
    VARNAME = "TEST_ENVVAR_ENUM_SET"
    CLI_ARGNAMES = ["--set"]
    CLI_HELP_MESSAGE = ""
    SPECIAL_SETTING_ENUM = SpecialEnvvarTestSetting

    @classmethod
    def default(cls):
        return {"x", SpecialEnvvarTestSetting.x}


def test_enum_envvar_set(monkeypatch):
    set1: set[str | SpecialEnvvarTestSetting] = {
        SpecialEnvvarTestSetting.y,
        "b",
        SpecialEnvvarTestSetting.z,
    }
    set2: set[str | SpecialEnvvarTestSetting] = {"x", "y", "z"}
    set3: set[str | SpecialEnvvarTestSetting] = {SpecialEnvvarTestSetting.x}

    TestEnumEnvvarSet.patchenv(monkeypatch, None)
    assert TestEnumEnvvarSet.get() == TestEnumEnvvarSet.default()
    assert TestEnumEnvvarSet.get(set1) == set1
    assert TestEnumEnvvarSet.parse(TestEnumEnvvarSet.serialize(set1)) == set1
    # Argparser with standard default
    parser = ArgumentParser()
    TestEnumEnvvarSet.add_arg(parser)
    assert parser.parse_args([]).set == TestEnumEnvvarSet.default()
    assert parser.parse_args(["--set", "x, 1 "]).set == TestEnumEnvvarSet.default()
    assert parser.parse_args(["--set", "x, y, z"]).set == set2
    # Passed-in default applies because the envvar is unset.
    parser = ArgumentParser()
    TestEnumEnvvarSet.add_arg(parser, default=set2)
    assert parser.parse_args([]).set == set2

    TestEnumEnvvarSet.patchenv(monkeypatch, "z, y, x")
    assert TestEnumEnvvarSet.get() == set2
    assert TestEnumEnvvarSet.parse(TestEnumEnvvarSet.serialize(set2)) == set2
    parser = ArgumentParser()
    TestEnumEnvvarSet.add_arg(parser)
    assert parser.parse_args([]).set == set2
    assert parser.parse_args(["--set", "1, x"]).set == TestEnumEnvvarSet.default()
    assert parser.parse_args(["--set", "y, z, x"]).set == set2

    TestEnumEnvvarSet.patchenv(monkeypatch, "1")
    assert TestEnumEnvvarSet.get() == set3
    assert TestEnumEnvvarSet.parse(TestEnumEnvvarSet.serialize(set3)) == set3
    # Argparser with standard default
    parser = ArgumentParser()
    TestEnumEnvvarSet.add_arg(parser)
    assert parser.parse_args([]).set == set3
    assert parser.parse_args(["--set", "1"]).set == set3
    assert parser.parse_args(["--set", "b, 2, 3"]).set == set1
    parser = ArgumentParser()
    # Passed-in default does not apply since the envvar is set.
    TestEnumEnvvarSet.add_arg(parser, default=set2)
    assert parser.parse_args([]).set == set3


class StringEnvvarWithDynamicDefault(QAIHMStringEnvvar):
    VARNAME = "TEST_STRING_DYNAMIC_DEFAULT"
    CLI_ARGNAMES = ["--str"]
    CLI_HELP_MESSAGE = ""
    DEFAULT_TEST_INT = 0

    @classmethod
    def default(cls):
        out = cls.DEFAULT_TEST_INT
        cls.DEFAULT_TEST_INT += 1
        return str(out)


def test_string_envvar_with_dynamic_default(monkeypatch):
    StringEnvvarWithDynamicDefault.DEFAULT_TEST_INT = 0
    StringEnvvarWithDynamicDefault.patchenv(monkeypatch, None)
    assert StringEnvvarWithDynamicDefault.get() == "0"
    assert StringEnvvarWithDynamicDefault.get() == "1"
