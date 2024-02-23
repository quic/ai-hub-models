# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Utility Functions for parsing input args for export and other customer facing scripts.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from functools import partial
from importlib import import_module
from pydoc import locate
from typing import Any, List, Mapping, Optional, Type

import qai_hub as hub

from qai_hub_models.utils.base_model import (
    BaseModel,
    FromPrecompiledTypeVar,
    FromPretrainedMixin,
    FromPretrainedTypeVar,
    InputSpec,
    TargetRuntime,
)
from qai_hub_models.utils.inference import HubModel
from qai_hub_models.utils.qai_hub_helpers import _AIHUB_NAME, can_access_qualcomm_ai_hub

DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S23"


class ParseEnumAction(argparse.Action):
    def __init__(self, option_strings, dest, enum_type, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.enum_type = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.enum_type[values.upper()])


def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def add_output_dir_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="If specified, saves demo output (e.g. image) to this directory instead of displaying.",
    )
    return parser


def add_target_runtime_arg(
    parser: argparse.ArgumentParser,
    help: str,
    default: TargetRuntime = TargetRuntime.TFLITE,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--target-runtime",
        type=str,
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore
        default=default,
        choices=[name.lower() for name in TargetRuntime._member_names_],
        help=help,
    )
    return parser


def get_on_device_demo_parser(
    parser: argparse.ArgumentParser | None = None,
    available_target_runtimes: List[TargetRuntime] = list(
        TargetRuntime.__members__.values()
    ),
    add_output_dir: bool = False,
):
    if not parser:
        parser = get_parser()

    parser.add_argument(
        "--on-device",
        action="store_true",
        help="If set, will evalute model using a Hub inference job instead of via torch.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="If running on-device, uses this model Hub model ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Samsung Galaxy S23",
        help="If running on-device, use this device.",
    )
    if add_output_dir:
        add_output_dir_arg(parser)
    parser.add_argument(
        "--device-os",
        type=str,
        default="",
        help="Optionally specified together with --device",
    )
    parser.add_argument(
        "--inference-options",
        type=str,
        default="",
        help="If running on-device, use these options when submitting the inference job.",
    )
    default_runtime = (
        TargetRuntime.TFLITE
        if TargetRuntime.TFLITE in available_target_runtimes
        else available_target_runtimes[0]
    )
    add_target_runtime_arg(
        parser,
        help="The runtime to demo (if --on-device is specified).",
        default=default_runtime,
    )

    return parser


def validate_on_device_demo_args(args: argparse.Namespace, model_name: str):
    """
    Validates the the args for the on device demo are valid.

    Intended for use only in CLI scripts.
    Prints error to console and exits if an error is found.
    """
    if args.on_device and not can_access_qualcomm_ai_hub():
        print(
            "On-device demos are not available without Qualcomm® AI Hub access.",
            "Please sign up for Qualcomm® AI Hub at https://aihub.qualcomm.com/.",
            sep=os.linesep,
        )
        sys.exit(1)

    if (args.inference_options or args.hub_model_id) and not args.on_device:
        print(
            "A Hub model ID and inference options can be provided only if the --on-device flag is provided."
        )
        sys.exit(1)


def get_model_cli_parser(
    cls: Type[FromPretrainedTypeVar], parser: argparse.ArgumentParser | None = None
) -> argparse.ArgumentParser:
    """
    Generate the argument parser to create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    if not parser:
        parser = get_parser()

    from_pretrained_sig = inspect.signature(cls.from_pretrained)
    for name, param in from_pretrained_sig.parameters.items():
        if name == "cls":
            continue
        # Determining type from param.annotation is non-trivial (it can be a
        # strings like "Optional[str]" or "bool | None").
        if param.default is not None:
            type_ = type(param.default)
        elif param.annotation == "bool":
            type_ = bool
        else:
            type_ = str
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type_,
            default=param.default,
            help=f"For documentation, see {cls.__name__}::from_pretrained.",
        )
    return parser


def get_model_kwargs(
    model_cls: Type[FromPretrainedTypeVar], args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model via `from_pretrained`.
    """
    from_pretrained_sig = inspect.signature(model_cls.from_pretrained)
    model_kwargs = {}
    for name in from_pretrained_sig.parameters:
        if name == "cls" or name not in args_dict:
            continue
        model_kwargs[name] = args_dict.get(name)
    return model_kwargs


def model_from_cli_args(
    model_cls: Type[FromPretrainedTypeVar], cli_args: argparse.Namespace
) -> FromPretrainedTypeVar:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    return model_cls.from_pretrained(**get_model_kwargs(model_cls, vars(cli_args)))


def demo_model_from_cli_args(
    model_cls: Type[FromPretrainedTypeVar],
    cli_args: argparse.Namespace,
    check_trace: bool = True,
) -> FromPretrainedTypeVar | HubModel:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.

    If the model is a BaseModel and an on-device demo is requested, the BaseModel will be wrapped in a HubModel.
    """
    model = model_from_cli_args(
        model_cls, cli_args
    )  # TODO(9494): This should be replaced by static input spec
    is_on_device = "on_device" in cli_args and cli_args.on_device
    inference_model: FromPretrainedTypeVar | HubModel
    if is_on_device and isinstance(model, BaseModel):
        device = hub.Device(cli_args.device, cli_args.device_os)
        if cli_args.hub_model_id:
            model_from_hub = hub.get_model(cli_args.hub_model_id)
            inference_model = HubModel(
                model_from_hub,
                list(model.get_input_spec().keys()),
                device,
                cli_args.inference_options,
            )
        else:
            model_cls = model_cls
            export_file = f"qai_hub_models.models.{model.get_model_id()}.export"
            export_module = import_module(export_file)
            compile_job: hub.CompileJob
            print(f"Compiling on-device model asset for {model.get_model_id()}.")
            print(
                f"Running python -m {export_file} --device {device.name} --target-runtime {cli_args.target_runtime.name.lower()}\n"
            )
            export_output = export_module.export_model(
                device=device.name,
                skip_profiling=True,
                skip_inferencing=True,
                skip_downloading=True,
                skip_summary=True,
                target_runtime=cli_args.target_runtime,
            )

            if len(export_output) == 0 or isinstance(export_output[0], str):
                # The export returned local file paths, which mean Hub credentials were not found.
                raise NotImplementedError(
                    f"Please sign-up for {_AIHUB_NAME} to continue the demo with on-device inference."
                )

            compile_job, _, _ = export_output
            target_model = compile_job.get_target_model()
            assert target_model is not None

            input_names = list(model.get_input_spec().keys())
            inference_model = HubModel(
                target_model,
                input_names,
                device,
                inference_options=cli_args.inference_options,
            )
            print(f"Exported asset: {inference_model.model.name}\n")
    else:
        inference_model = model
    return inference_model


def get_input_spec_kwargs(
    model: "BaseModel", args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model's input_spec.
    """
    get_input_spec_args = inspect.signature(model.get_input_spec)
    input_spec_kwargs = {}
    for name in get_input_spec_args.parameters:
        if name == "self" or name not in args_dict:
            continue
        input_spec_kwargs[name] = args_dict[name]
    return input_spec_kwargs


def get_model_input_spec_parser(
    model_cls: Type[BaseModel], parser: argparse.ArgumentParser | None = None
) -> argparse.ArgumentParser:
    """
    Generate the argument parser to get this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    """
    if not parser:
        parser = get_parser()

    get_input_spec_sig = inspect.signature(model_cls.get_input_spec)
    for name, param in get_input_spec_sig.parameters.items():
        if name == "self":
            continue
        type_: type | object
        if isinstance(param.annotation, type):
            type_ = param.annotation
        else:
            # locate() converts string type to cls type
            # Any type can be resolved as long as it's accessible in this scope
            type_ = locate(param.annotation)
            assert isinstance(type_, type)
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type_,
            default=param.default,
            help=f"For documentation, see {model_cls.__name__}::get_input_spec.",
        )
    return parser


def input_spec_from_cli_args(
    model: "BaseModel", cli_args: argparse.Namespace
) -> "InputSpec":
    """
    Create this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    """
    return model.get_input_spec(**get_input_spec_kwargs(model, vars(cli_args)))


def export_parser(
    model_cls: Type[FromPretrainedTypeVar] | Type[FromPrecompiledTypeVar],
    components: Optional[List[str]] = None,
    supports_qnn=True,
    exporting_compiled_model=False,
) -> argparse.ArgumentParser:
    """
    Arg parser to be used in export scripts.

    Parameters:
        model_cls: Class of the model to be exported. Used to add additional
            args for model instantiation.
        components: Some models have multiple components that need to be
            compiled separately. This represents the list of options for the user to
            select which components they want to compile.
        supports_qnn:
            Whether QNN export is supported.
            Default=True.
        exporting_compiled_model:
            True when exporting compiled model.
            If set, removing skip_profiling flag from export arguments.
            Default = False.

    Returns:
        Arg parser object.
    """
    parser = get_parser()
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_EXPORT_DEVICE,
        help="Device for which to export.",
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="If set, writes compiled model to local directory without profiling.",
    )
    parser.add_argument(
        "--skip-inferencing",
        action="store_true",
        help="If set, skips verifying on-device output vs local cpu.",
    )
    if not exporting_compiled_model:
        parser.add_argument(
            "--skip-downloading",
            action="store_true",
            help="If set, skips downloading of compiled model.",
        )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="If set, skips printing summary of inference and profiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated assets (e.g. compiled model). "
        "Defaults to `<cwd>/build/<model_name>`.",
    )
    if not exporting_compiled_model:
        # Default runtime for compiled model is fixed for given model
        add_target_runtime_arg(parser, help="The runtime for which to export.")
        # No compilation for compiled models
        parser.add_argument(
            "--compile-options",
            type=str,
            default="",
            help="Additional options to pass when submitting the compile job.",
        )
    parser.add_argument(
        "--profile-options",
        type=str,
        default="",
        help="Additional options to pass when submitting the profile job.",
    )
    if components is not None:
        parser.add_argument(
            "--components",
            nargs="+",
            type=str,
            default=None,
            choices=components,
            help="Which components of the model to be exported.",
        )

    if issubclass(model_cls, FromPretrainedMixin):
        # Skip adding CLI from model for compiled model
        # TODO: #9408 Refactor BaseModel, BasePrecompiledModel to fetch
        # parameters from compiled model
        parser = get_model_cli_parser(model_cls, parser)

        if issubclass(model_cls, BaseModel):
            parser = get_model_input_spec_parser(model_cls, parser)

    return parser
