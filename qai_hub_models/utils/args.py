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
from pydoc import locate
from typing import Any, List, Mapping, Optional, Set, Type

import qai_hub as hub
from qai_hub.client import APIException, UserError

from qai_hub_models.models.protocols import (
    FromPrecompiledTypeVar,
    FromPretrainedProtocol,
    FromPretrainedTypeVar,
)
from qai_hub_models.utils.base_model import BaseModel, HubModel, TargetRuntime
from qai_hub_models.utils.inference import OnDeviceModel, compile_model_from_args
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub

DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S23 (Family)"


class ParseEnumAction(argparse.Action):
    def __init__(self, option_strings, dest, enum_type, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.enum_type = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.enum_type[values.upper().replace("-", "_")])


def get_parser(allow_dupe_args: bool = False) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve" if allow_dupe_args else "error",
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


def _get_default_runtime(available_runtimes: List[TargetRuntime]):
    if len(available_runtimes) == 0:
        raise RuntimeError("available_runtimes empty, expecting at-least one runtime.")

    return (
        TargetRuntime.TFLITE
        if TargetRuntime.TFLITE in available_runtimes
        else available_runtimes[0]
    )


def add_target_runtime_arg(
    parser: argparse.ArgumentParser,
    help: str,
    available_target_runtimes: List[TargetRuntime] = list(
        TargetRuntime.__members__.values()
    ),
    default: TargetRuntime = TargetRuntime.TFLITE,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--target-runtime",
        type=str,
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore
        default=default,
        choices=[rt.name.lower().replace("_", "-") for rt in available_target_runtimes],
        help=help,
    )
    return parser


def get_on_device_demo_parser(
    parser: argparse.ArgumentParser | None = None,
    available_target_runtimes: List[TargetRuntime] = list(
        TargetRuntime.__members__.values()
    ),
    add_output_dir: bool = False,
    default_device: str = "Samsung Galaxy S23",
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
        help="If running on-device, uses this model Hub model ID."
        " Provide comma separated model-ids if multiple models are required for demo."
        " Run export.py to get on-device demo command with models exported for you.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="If running on-device, use this device.",
    )
    parser.add_argument(
        "--chipset",
        type=str,
        default=None,
        choices=sorted(get_qcom_chipsets(), reverse=True),
        help="If set, will choose a random device with this chipset. "
        "Overrides whatever is set in --device.",
    )
    parser.add_argument(
        "--device-os",
        type=str,
        default="",
        help="Optionally specified together with --device",
    )
    if add_output_dir:
        add_output_dir_arg(parser)
    parser.add_argument(
        "--inference-options",
        type=str,
        default="",
        help="If running on-device, use these options when submitting the inference job.",
    )
    default_runtime = _get_default_runtime(available_runtimes=available_target_runtimes)
    add_target_runtime_arg(
        parser,
        help="The runtime to demo (if --on-device is specified).",
        default=default_runtime,
        available_target_runtimes=available_target_runtimes,
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
            "Please sign up for Qualcomm® AI Hub at https://myaccount.qualcomm.com/signup .",
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

        help = (
            f"For documentation, see {cls.__name__}::from_pretrained::parameter {name}."
        )

        # Determining type from param.annotation is non-trivial (it can be a
        # strings like "Optional[str]" or "bool | None").
        bool_action = None
        arg_name = f"--{name.replace('_', '-')}"
        if param.default is not None:
            type_ = type(param.default)

            if type_ == bool:
                if param.default:
                    bool_action = "store_false"
                    # If the default is true, and the arg name does not start with no_,
                    # then add the no- to the argument (as it should be passed as --no-enable-flag, not --enable-flag)
                    if name.startswith("no_"):
                        arg_name = f"--{name[3:].replace('_', '-')}"
                    else:
                        arg_name = f"--no-{name.replace('_', '-')}"
                    help = (
                        f"{help} Setting this flag will set parameter {name} to False."
                    )
                else:
                    bool_action = "store_true"
                    # If the default is false, and the arg name starts with no_,
                    # then remove the no- from the argument (as it should be passed as --enable-flag, not --no-enable-flag)
                    arg_name = f"--{name.replace('_', '-')}"
                    help = (
                        f"{help} Setting this flag will set parameter {name} to True."
                    )
        elif param.annotation == "bool":
            type_ = bool
        else:
            type_ = str

        if bool_action:
            parser.add_argument(arg_name, dest=name, action=bool_action, help=help)
        else:
            parser.add_argument(
                arg_name,
                dest=name,
                type=type_,
                default=param.default,
                help=help,
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


def get_hub_device(
    device: Optional[str] = None, chipset: Optional[str] = None, device_os: str = ""
) -> hub.Device:
    """
    Get a hub.Device given a device name or chipset name.
    If chipset is specified, that takes precedence over the device name.
    If neither is specified, the function throws an error.
    """
    if chipset:
        return hub.Device(attributes=f"chipset:{chipset}", os=device_os)
    if device:
        return hub.Device(name=device, os=device_os)
    raise ValueError("Must specify one of device or chipset")


def demo_model_from_cli_args(
    model_cls: Type[FromPretrainedTypeVar],
    model_id: str,
    cli_args: argparse.Namespace,
) -> FromPretrainedTypeVar | OnDeviceModel:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.

    If the model is a BaseModel and an on-device demo is requested,
        the BaseModel will be wrapped in an OnDeviceModel.
    """
    is_on_device = "on_device" in cli_args and cli_args.on_device
    inference_model: FromPretrainedTypeVar | OnDeviceModel
    if is_on_device and issubclass(model_cls, BaseModel):
        device = get_hub_device(cli_args.device, cli_args.chipset, cli_args.device_os)
        if cli_args.hub_model_id:
            model_from_hub = hub.get_model(cli_args.hub_model_id)
            inference_model = OnDeviceModel(
                model_from_hub,
                list(model_cls.get_input_spec().keys()),
                device,
                cli_args.inference_options,
            )
        else:
            cli_dict = vars(cli_args)
            additional_kwargs = dict(
                get_model_kwargs(model_cls, cli_dict),
                **get_input_spec_kwargs(model_cls, cli_dict),
            )
            target_model = compile_model_from_args(
                model_id, cli_args, additional_kwargs
            )
            input_names = list(model_cls.get_input_spec().keys())
            inference_model = OnDeviceModel(
                target_model,
                input_names,
                device,
                inference_options=cli_args.inference_options,
            )
            print(f"Exported asset: {model_id}\n")

    else:
        inference_model = model_from_cli_args(model_cls, cli_args)
    return inference_model


def get_input_spec_kwargs(
    model: HubModel | Type[HubModel], args_dict: Mapping[str, Any]
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
    model: HubModel | OnDeviceModel, cli_args: argparse.Namespace
) -> hub.InputSpecs:
    """
    Create this model's input spec from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as get_input_spec method args.
    Also, fetches shapes if demo is run on-device.
    """
    if isinstance(model, OnDeviceModel):
        assert "on_device" in cli_args and cli_args.on_device
        assert isinstance(model.model.producer, hub.CompileJob)
        return model.model.producer.shapes
    return model.get_input_spec(**get_input_spec_kwargs(model, vars(cli_args)))


def get_qcom_chipsets() -> Set[str]:
    try:
        return set(
            [
                attr[len("chipset:") :]
                for dev in hub.get_devices()
                for attr in dev.attributes
                if attr.startswith("chipset:qualcomm")
            ]
        )
    except (APIException, UserError):
        return set([])


def _evaluate_export_common_parser(
    model_cls: Type[FromPretrainedTypeVar] | Type[FromPrecompiledTypeVar],
    supports_tflite=True,
    supports_qnn=True,
    supports_onnx=True,
    supports_precompiled_qnn_onnx=True,
    default_runtime=TargetRuntime.TFLITE,
    exporting_compiled_model=False,
) -> argparse.ArgumentParser:
    """
    Common arguments between export and evaluate scripts.
    """
    # Set handler to resolve, to allow from_pretrained and get_input_spec
    # to have the same argument names.
    parser = get_parser(allow_dupe_args=True)

    if not exporting_compiled_model:
        # Default runtime for compiled model is fixed for given model
        available_runtimes = []
        if supports_tflite:
            available_runtimes.append(TargetRuntime.TFLITE)
        if supports_qnn:
            available_runtimes.append(TargetRuntime.QNN)
        if supports_onnx:
            available_runtimes.append(TargetRuntime.ONNX)
        if supports_precompiled_qnn_onnx:
            available_runtimes.append(TargetRuntime.PRECOMPILED_QNN_ONNX)

        default_runtime = _get_default_runtime(available_runtimes)
        add_target_runtime_arg(
            parser,
            available_target_runtimes=available_runtimes,
            default=default_runtime,
            help="The runtime for which to export.",
        )
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
    if issubclass(model_cls, FromPretrainedProtocol):
        # Skip adding CLI from model for compiled model
        # TODO: #9408 Refactor BaseModel, BasePrecompiledModel to fetch
        # parameters from compiled model
        parser = get_model_cli_parser(model_cls, parser)

        if issubclass(model_cls, BaseModel):
            parser = get_model_input_spec_parser(model_cls, parser)

    return parser


def export_parser(
    model_cls: Type[FromPretrainedTypeVar] | Type[FromPrecompiledTypeVar],
    components: Optional[List[str]] = None,
    supports_tflite: bool = True,
    supports_qnn: bool = True,
    supports_onnx: bool = True,
    supports_precompiled_qnn_onnx: bool = True,
    default_runtime: TargetRuntime = TargetRuntime.TFLITE,
    exporting_compiled_model: bool = False,
    default_export_device: str = DEFAULT_EXPORT_DEVICE,
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
        supports_onnx:
            Whether ORT export is supported.
            Default=True.
        supports_precompiled_qnn_onnx:
            Whether precompiled ORT (with QNN context binary) export is supported.
            Default=True.
        default_runtime: Which runtime to use as default if not specified in cli args.
        exporting_compiled_model:
            True when exporting compiled model.
            If set, removing skip_profiling flag from export arguments.
            Default = False.
        default_export_device:
            Default device to set for export.

    Returns:
        Arg parser object.
    """
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supports_tflite=supports_tflite,
        supports_qnn=supports_qnn,
        supports_onnx=supports_onnx,
        supports_precompiled_qnn_onnx=supports_precompiled_qnn_onnx,
        default_runtime=default_runtime,
        exporting_compiled_model=exporting_compiled_model,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_export_device,
        help="Device for which to export.",
    )
    parser.add_argument(
        "--chipset",
        type=str,
        default=None,
        choices=sorted(get_qcom_chipsets(), reverse=True),
        help="If set, will choose a random device with this chipset. "
        "Overrides whatever is set in --device.",
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
    if components is not None:
        parser.add_argument(
            "--components",
            nargs="+",
            type=str,
            default=None,
            choices=components,
            help="Which components of the model to be exported.",
        )
    return parser


def evaluate_parser(
    model_cls: Type[FromPretrainedTypeVar] | Type[FromPrecompiledTypeVar],
    default_split_size: int,
    supported_datasets: List[str],
    supports_tflite=True,
    supports_qnn=True,
    supports_onnx=True,
    default_runtime=TargetRuntime.TFLITE,
) -> argparse.ArgumentParser:
    """
    Arg parser to be used in evaluate scripts.

    Parameters:
        model_cls: Class of the model to be exported. Used to add additional
            args for model instantiation.
        supported_datasets: List of supported dataset names.
        default_split_size: Default size for the most samples to be submitted
            in a single inference job.
        supports_qnn:
            Whether QNN export is supported.
            Default=True.
        supports_onnx:
            Whether ORT export is supported.
            Default=True.
        exporting_compiled_model:
            True when exporting compiled model.
            If set, removing skip_profiling flag from export arguments.
            Default = False.
        default_runtime: Which runtime to use as default if not specified in cli args.

    Returns:
        Arg parser object.
    """
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supports_tflite=supports_tflite,
        supports_qnn=supports_qnn,
        supports_onnx=supports_onnx,
        default_runtime=default_runtime,
    )
    parser.add_argument(
        "--chipset",
        type=str,
        default="qualcomm-snapdragon-8gen2",
        choices=sorted(get_qcom_chipsets(), reverse=True),
        help="Which chipset to use to run evaluation.",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=default_split_size,
        help="Max size to be submitted in a single inference job.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=supported_datasets[0],
        choices=supported_datasets,
        help="Name of the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to run. If set to -1, will run on full dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use when shuffling the data.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="A compiled hub model id.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="If set, will store hub dataset ids in a local file and re-use "
        "for subsequent evaluations on the same dataset.",
    )
    return parser
