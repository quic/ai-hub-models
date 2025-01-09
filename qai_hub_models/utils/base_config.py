# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from types import GenericAlias
from typing import Any, Optional, TypeVar, Union, get_args, get_type_hints

import yaml
from schema import And
from schema import Optional as OptionalSchema
from schema import Schema

from qai_hub_models.utils.asset_loaders import load_yaml


def _get_origin(input_type: type) -> type:
    """
    For nested types like list[str] or Union[str, int], this function will
        return the "parent" type like List or Union.

    If the input type is not a nested type, the function returns the input_type.
    """
    return getattr(input_type, "__origin__", input_type)


def _extract_optional_type(input_type: type) -> type:
    """
    Given an optional type as input, returns the inner type that is wrapped.

    For example, if input type is Optional[int], the function returns int.
    """
    assert (
        _get_origin(input_type) == Union
    ), "Input type must be an instance of `Optional`."
    union_args = get_args(input_type)
    assert len(union_args) == 2 and issubclass(
        union_args[1], type(None)
    ), "Input type must be an instance of `Optional`."
    return union_args[0]


def _constructor_from_type(input_type: type) -> Union[type, Callable]:
    """
    Given a type, return the appropriate constructor for that type.

    For primitive types like str and int, the type and constructor are the same object.

    For types like List, the constructor is list.
    """
    if input_type == GenericAlias:
        return input_type
    input_type = _get_origin(input_type)
    if input_type == list:
        return list
    if input_type == dict:
        return dict
    return input_type


@dataclass
class BaseQAIHMConfig:
    """
    A base class for all AI Hub Models configs.
    Config fields are defined as typed dataclass fields.

    This class is capable of loading a YAML file (via .from_yaml()) or
    an arbirary python dict (via .from_dict()) into an instance of itself.

    The class instance is also capable of dumping itself to a dictionary
    or to yaml (via .to_yaml() or .to_dict()).
    """

    @classmethod
    def get_schema(cls) -> Schema:
        """Derive the Schema from the fields set on the dataclass."""
        schema_dict = {}
        field_datatypes = get_type_hints(cls)
        for field in fields(cls):
            field_type = field_datatypes[field.name]
            if _get_origin(field_type) == Union:
                field_type = _extract_optional_type(field_type)
                assert (
                    field.default != dataclasses.MISSING
                ), "Optional fields must have a default set."
            if field.default != dataclasses.MISSING:
                field_key = OptionalSchema(field.name, default=field.default)
            elif field.default_factory != dataclasses.MISSING:
                field_key = OptionalSchema(field.name, default=field.default_factory())
            else:
                field_key = field.name
            schema_dict[field_key] = _constructor_from_type(field_type)
        return Schema(And(schema_dict))

    @classmethod
    def from_dict(
        cls: type[BaseQAIHMConfigTypeVar], val_dict: dict[str, Any]
    ) -> BaseQAIHMConfigTypeVar:
        """
        Reads the dict at the given path and loads it into an instance of this class.

        The input dictionary may be modified by this function. It should be copied
        if the user wants to keep a copy of it in its original state.
        """
        # Validate schema
        val_dict = cls.get_schema().validate(val_dict)

        # Build dataclass
        kwargs = {field.name: val_dict[field.name] for field in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def from_yaml(
        cls: type[BaseQAIHMConfigTypeVar], path: str | Path
    ) -> BaseQAIHMConfigTypeVar:
        """
        Reads the yaml file at the given path and loads it into an instance of this class.
        """
        return cls.from_dict(load_yaml(path))

    def validate(self) -> Optional[str]:
        """
        Returns a string reason if the this class is not valid, or None if it is valid.
        """
        return None

    def to_dict(
        self, include_defaults: bool = True, yaml_compatible=False
    ) -> dict[str, Any]:
        """
        Returns this class as a python dictionary.

        parameters:
            include_defaults : bool
                If false, dataclass fields will not be included in the dict if set to the fields' default value.

            yaml_compatible : bool
                Returns a dict in which all Python objects are converted to a YAML-serializable representation.
        """
        return self._complete_partial_dict(
            include_defaults=include_defaults, yaml_compatible=yaml_compatible
        )

    def _complete_partial_dict(
        self,
        partial_yaml_dict: Optional[dict] = None,
        include_defaults: bool = True,
        yaml_compatible: bool = False,
    ):
        """
        Fills partial_yaml_dict with all fields of the dataclass that do not exist in the dict.

        parameters:
            partial_yaml_dict : Optional[dict]
                The dict to fill. If unset, uses a new empty dict.

            include_defaults : bool
                If false, dataclass fields will not be included in the dict if set to the fields' default value.

            yaml_compatible : bool
                Returns a dict in which all Python objects are converted to a YAML-serializable representation.

        discussion:
            This function should be used after to_dict() processes complex values and adds them to the dict.
            For example, to_dict() may fill a dict with a special string representation of enum fields.
            After that, it passes that dict to this func to naively dump all other fields in the class into the dictionary.
        """

        def _process_dict_field_val(field_val: dict[Any, Any]):
            out_dict = {}
            for k, v in field_val.items():
                out_dict[_process_field_val(k)] = _process_field_val(v)
            return out_dict

        def _process_list_field_val(field_val: list[Any]):
            out_list = []
            for val in field_val:
                out_list.append(_process_field_val(val))
            return out_list

        def _process_tuple_field_val(field_val: tuple[Any, ...]):
            return tuple(_process_list_field_val(list(field_val)))

        def _process_field_val(field_val: Any):
            if isinstance(field_val, dict):
                return _process_dict_field_val(field_val)
            elif isinstance(field_val, list):
                return _process_list_field_val(field_val)
            elif isinstance(field_val, tuple):
                return _process_tuple_field_val(field_val)
            elif isinstance(field_val, BaseQAIHMConfig):
                return field_val.to_dict(include_defaults)
            elif yaml_compatible and type(field_val) not in [int, float, bool, str]:
                return str(field_val)
            return field_val

        fields = dataclasses.fields(self)
        yaml_dict = partial_yaml_dict or {}
        for field in fields:
            default = field.default if isinstance(field.default_factory, dataclasses._MISSING_TYPE) else field.default_factory()  # type: ignore
            field_val = getattr(self, field.name)
            if field.name not in yaml_dict and (
                include_defaults or field_val != default
            ):
                yaml_dict[field.name] = _process_field_val(field_val)

        return yaml_dict

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = True,
        delete_if_empty: bool = True,
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

        discussion:
            Generally, the dictionary to be saved to YAML is empty only if:
             * all dataclass fields have default values
             * every field in this dataclass instance is set to its default value
        """
        dict = self.to_dict(include_defaults=False, yaml_compatible=True)
        if not dict and not write_if_empty:
            if delete_if_empty and os.path.exists(path):
                os.remove(path)
            return False
        with open(path, "w") as yaml_file:
            yaml.dump(dict, yaml_file)
        return True


BaseQAIHMConfigTypeVar = TypeVar("BaseQAIHMConfigTypeVar", bound=BaseQAIHMConfig)
