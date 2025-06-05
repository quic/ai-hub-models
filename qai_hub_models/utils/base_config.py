# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Generic

import ruamel.yaml
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from ruamel.yaml.representer import RoundTripRepresenter
from typing_extensions import TypeVar


class BaseQAIHMConfig(BaseModel):
    """
    A base class for all AI Hub Models configs.
    Config fields are defined as typed dataclass fields.

    This class is capable of loading a YAML file (via .from_yaml()) or
    an arbirary python dict (via .from_dict()) into an instance of itself.

    The class instance is also capable of dumping itself to a dictionary
    or to yaml (via .to_yaml() or .to_dict()).
    """

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        **kwargs,
    ) -> bool:
        """
        Converts this class to a dict and saves that dict to a YAML file.

        parameters:
            path : str | Path
                Path to save the file.

            write_if_empty : bool
                If False, the YAML file will not be written to disk if the dictionary to be saved is empty.

            delete_if_empty: bool
                If True, an existing YAML file at the given path will be deleted if the dictionary to be saved is empty.

            **kwargs
                Additional args (used by overrides).

        discussion:
            Generally, the dictionary to be saved to YAML is empty only if:
             * all dataclass fields have default values
             * every field in this dataclass instance is set to its default value
        """
        yaml = ruamel.yaml.YAML()

        # build_and_test.py uses simplistic YAML readers that can't read strings on multiple lines.
        # Make sure strings aren't dumped to multiple lines in the YAML.
        yaml.width = 4096

        # Allow strings with newlines to dump as newlines rather than \n
        def _yaml_repr_str(dumper: RoundTripRepresenter, data: str):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.representer.add_representer(str, _yaml_repr_str)

        # Dump data
        to_yaml_file(
            path,
            self,
            custom_yaml_writer=yaml,
            exclude_defaults=True,
            exclude_none=True,
            **kwargs,
        )

        # Remove file if empty
        if (not write_if_empty or delete_if_empty) and os.path.getsize(path) == 0:
            os.remove(path)
            return False

        return True

    @classmethod
    def from_yaml(
        cls: type[BaseQAIHMConfigTypeVar],
        path: str | Path,
        create_empty_if_no_file: bool = False,
    ) -> BaseQAIHMConfigTypeVar:
        """
        Reads the yaml file at the given path and loads it into an instance of this class.
        """
        if create_empty_if_no_file and (
            not os.path.exists(path) or os.path.getsize(path) == 0
        ):
            return cls()
        return parse_yaml_file_as(cls, path)


BaseQAIHMConfigTypeVar = TypeVar("BaseQAIHMConfigTypeVar", bound=BaseQAIHMConfig)


EnumT = TypeVar("EnumT", bound=Enum)


class EnumListWithParseableAll(list[EnumT], Generic[EnumT]):
    """
    Helper list class that can parse an enum list to / from "all".
    If "all" is in the list, then all enum elements are returned.
    """

    # Subclasses should set this to the EnumT class.
    EnumType: type[Enum]
    ALL: list[EnumT] | None = None

    @classmethod
    def default(
        cls: type[EnumListWithParseableAllTypeVar],
    ) -> EnumListWithParseableAllTypeVar:
        if cls.ALL is not None:
            return cls(cls.ALL)
        return cls([x for x in cls.EnumType])

    @classmethod
    def parse(
        cls: type[EnumListWithParseableAllTypeVar], obj: Any
    ) -> EnumListWithParseableAllTypeVar:
        if isinstance(obj, list):
            out: EnumListWithParseableAllTypeVar = cls()
            for x in obj:
                if x == "all":
                    out = cls.default()
                else:
                    out.append(cls.EnumType(x))
            return out
        raise ValueError(f"Unsupported type {type(obj)} for parsing to {cls}")

    @classmethod
    def serialize(
        cls: type[EnumListWithParseableAllTypeVar], list: list[EnumT]
    ) -> list[str]:
        if len(set(list)) == len(cls.EnumType):
            return ["all"]
        return [x.value for x in list]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            field_name=handler.field_name,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.serialize, when_used="json"
            ),
        )


EnumListWithParseableAllTypeVar = TypeVar(
    "EnumListWithParseableAllTypeVar", bound=EnumListWithParseableAll
)
