# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Utility Functions for parsing input args for export and other customer facing scripts.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import os
import sys
from collections.abc import Mapping
from functools import partial
from pydoc import locate
from typing import Any, Optional, TypeVar

import qai_hub as hub
import torch
from qai_hub.client import APIException, InternalError, UserError

from qai_hub_models.models.common import Precision
from qai_hub_models.models.protocols import (
    FromPrecompiledTypeVar,
    FromPretrainedProtocol,
    FromPretrainedTypeVar,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    HubModel,
    TargetRuntime,
)
from qai_hub_models.utils.evaluate import EvalMode
from qai_hub_models.utils.inference import OnDeviceModel, compile_model_from_args
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub


class ParseEnumAction(argparse.Action):
    def __init__(self, option_strings, dest, enum_type, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.enum_type = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str)
        setattr(namespace, self.dest, self.enum_type[values.upper().replace("-", "_")])


ParserT = TypeVar("ParserT", bound=argparse.ArgumentParser)


def _get_non_float_precision(
    supported_precisions: set[Precision] | None,
) -> Precision | None:
    if not supported_precisions:
        return None

    for p in supported_precisions:
        if p != Precision.float:
            return p

    return None


def get_quantize_action_with_default(
    default_quantized_precision: Precision,
) -> type[argparse.Action]:
    """
    Get an action that:

    Returns default_quantized_precision if "--quantize" is passed with no arg.

    Returns a parsed precision object if "--quantize <value> " is passed.
    """

    class ParsePrecisionAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if values:
                if isinstance(values, Precision):
                    val = values
                else:
                    assert isinstance(values, str)
                    val = Precision.parse(values)
            else:
                val = default_quantized_precision

            setattr(namespace, self.dest, val)

    return ParsePrecisionAction


class QAIHMArgumentParser(argparse.ArgumentParser):
    """
    An ArgumentParser that sets hub_device from the appropriate options.
    This isn't implemented as a `type` argument to `add_argument` because the
    device/chipset can be modified by device_os.
    """

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        device_name = getattr(parsed, "device", None)
        chipset_name = getattr(parsed, "chipset", None)
        if device_name or chipset_name:
            hub_device = _get_hub_device(
                device_name, chipset_name, getattr(parsed, "device_os", "")
            )
            parsed.hub_device = hub_device

        if getattr(parsed, "quantize", None):
            parsed.precision = parsed.quantize

        return parsed


def get_parser(allow_dupe_args: bool = False) -> QAIHMArgumentParser:
    return QAIHMArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve" if allow_dupe_args else "error",
    )


def _add_device_args(
    parser: QAIHMArgumentParser,
    default_device: str | None = None,
    default_chipset: str | None = None,
) -> QAIHMArgumentParser:
    # This is an assertion because this is a logic error; it shouldn't be possible to get this at runtime.
    assert not (
        default_device and default_chipset
    ), "Only one of default_device or default_chipset may be specified."

    device_group = parser.add_argument_group("Device Selection")
    device_mutex_group = device_group.add_mutually_exclusive_group()
    device_mutex_group.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="If running on-device, use this device.",
    )
    device_mutex_group.add_argument(
        "--chipset",
        type=str,
        default=default_chipset,
        choices=sorted(_get_qcom_chipsets(), reverse=True),
        help="If set, will choose a random device with this chipset.",
    )
    device_group.add_argument(
        "--device-os",
        type=str,
        default="",
        help="Optionally specified together with --device or --chipset",
    )

    return parser


def add_output_dir_arg(parser: ParserT) -> ParserT:
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="If specified, saves demo output (e.g. image) to this directory instead of displaying.",
    )
    return parser


def _get_default_runtime(available_runtimes: list[TargetRuntime] | set[TargetRuntime]):
    if len(available_runtimes) == 0:
        raise RuntimeError("available_runtimes empty, expecting at-least one runtime.")

    return (
        TargetRuntime.TFLITE
        if TargetRuntime.TFLITE in available_runtimes
        else next(iter(available_runtimes))
    )


def add_target_runtime_arg(
    parser: ParserT,
    help: str,
    available_target_runtimes: list[TargetRuntime]
    | set[TargetRuntime] = set(TargetRuntime.__members__.values()),
    default: TargetRuntime = TargetRuntime.TFLITE,
) -> ParserT:
    parser.add_argument(
        "--target-runtime",
        type=str,
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore[arg-type]
        default=default,
        choices=[rt.value for rt in available_target_runtimes],
        help=help,
    )
    return parser


def add_precision_arg(
    parser: argparse.ArgumentParser,
    supported_precisions: set[Precision],
    default_if_arg_explicitly_passed: Precision,  # the default value if --precision is passed explicitly
    default: Precision,  # the default value if --precision is not passed
) -> argparse.ArgumentParser:
    precision_help = "Desired precision to which the model should be quantized."
    if Precision.float in supported_precisions:
        precision_help += " If set to 'float', the model will not be quantized, and inference will run in fp32 or fp16 (depending on compute unit)."

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--precision",
        action=get_quantize_action_with_default(default),
        default=default,
        choices=[str(p) for p in supported_precisions],
        help=precision_help,
    )
    if len(supported_precisions) > 1:
        group.add_argument(
            "--quantize",
            action=get_quantize_action_with_default(default_if_arg_explicitly_passed),
            default=None,
            choices=[str(p) for p in supported_precisions if p != Precision.float],
            help=f"Quantize the model to this precision. If passed without an explicit argument, precision {default_if_arg_explicitly_passed} will be used. If set, this always supercedes the '--precision' argument.",
            nargs="?",
        )
    return parser


def get_on_device_demo_parser(
    parser: QAIHMArgumentParser | None = None,
    supported_eval_modes: list[EvalMode] | None = None,
    supported_precisions: set[Precision] | None = None,
    available_target_runtimes: list[TargetRuntime]
    | set[TargetRuntime] = set(TargetRuntime.__members__.values()),
    add_output_dir: bool = False,
    default_device: str | None = None,
    default_host_device: torch.device = torch.device("cpu"),
):
    """
    Args:
    - supported_eval_modes: subset of
    EvalMode.{FP,QUANTSIM,ON_DEVICE,LOCAL_DEVICE}. Default is
    [EvalMode.FP, EvalMode.ON_DEVICE]. The first value of supported_eval_modes
    will be the default.

    - supported_precisions: subset of {Precision.float, Precision.w8a8,
    Precision.w8a16}
    """
    if not parser:
        parser = get_parser()

    # Add --eval-mode
    supported_eval_modes = supported_eval_modes or [EvalMode.FP, EvalMode.ON_DEVICE]

    # take the first allowed mode as the default
    default_mode = supported_eval_modes[0]

    mode_help_lines = ["Run the model in one of the following modes:"]
    for m in supported_eval_modes:
        mode_help_lines.append(f"  - {m.value}: {m.description}")
    mode_help_msg = "\n".join(mode_help_lines)

    parser.add_argument(
        "--eval-mode",
        type=EvalMode.from_string,
        choices=supported_eval_modes,
        default=default_mode,
        help=mode_help_msg,
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="If mode==on-device, uses this model Hub model ID."
        " Provide comma separated model-ids if multiple models are required for demo."
        " Run export.py to get on-device demo command with models exported for you.",
    )

    _add_device_args(parser, default_device=default_device)

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
        help="The runtime to demo (if `--eval-mode on-device` is specified).",
        default=default_runtime,
        available_target_runtimes=available_target_runtimes,
    )

    # TODO: This should only include supported precisions.
    default_precisions = {Precision.float, Precision.w8a8, Precision.w8a16}
    supported_precisions = supported_precisions or default_precisions
    add_precision_arg(
        parser,
        supported_precisions,
        list(supported_precisions)[0],
        list(supported_precisions)[0],
    )

    return parser


def validate_on_device_demo_args(args: argparse.Namespace, model_name: str):
    """
    Validates the the args for the on device demo are valid.

    Intended for use only in CLI scripts.
    Prints error to console and exits if an error is found.
    """
    is_on_device = args.eval_mode == EvalMode.ON_DEVICE
    if is_on_device and not can_access_qualcomm_ai_hub():
        print(
            "On-device demos (--eval-mode on-device) are not available without Qualcomm® AI Hub access.",
            "Please sign up for Qualcomm® AI Hub at https://myaccount.qualcomm.com/signup .",
            sep=os.linesep,
        )
        sys.exit(1)

    if is_on_device and not getattr(args, "hub_device", None):
        print("--device or --chipset must be specified with --eval-mode on-device.")
        sys.exit(1)

    if (args.inference_options or args.hub_model_id) and not is_on_device:
        print(
            "A Hub model ID and inference options can be provided only with --eval-mode on-device."
        )
        sys.exit(1)


def get_model_cli_parser(
    cls: type[FromPretrainedTypeVar],
    parser: QAIHMArgumentParser | None = None,
    suppress_help_arguments: list | None = None,
) -> QAIHMArgumentParser:
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

        def get_help(arg_name: str) -> str:
            # Suppress help for argument that need not be exposed for model.
            if suppress_help_arguments is not None:
                if arg_name in suppress_help_arguments:
                    return argparse.SUPPRESS
            return help

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
            parser.add_argument(
                arg_name, dest=name, action=bool_action, help=get_help(arg_name)
            )
        else:
            parser.add_argument(
                arg_name,
                dest=name,
                type=type_,
                default=param.default,
                help=get_help(arg_name),
            )
    return parser


def get_model_kwargs(
    model_cls: type[FromPretrainedTypeVar], args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model via `from_pretrained`.
    """
    from_pretrained_sig = inspect.signature(model_cls.from_pretrained)
    model_kwargs = {}
    for name in from_pretrained_sig.parameters:
        if name in ["cls", "kwargs"] or name not in args_dict:
            continue
        model_kwargs[name] = args_dict.get(name)
    return model_kwargs


def model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar], cli_args: argparse.Namespace
) -> FromPretrainedTypeVar:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    return model_cls.from_pretrained(**get_model_kwargs(model_cls, vars(cli_args)))


def _get_hub_device(
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


def demo_model_components_from_cli_args(
    model_cls: type[CollectionModel],
    model_id: str,
    cli_args: argparse.Namespace,
) -> tuple[FromPretrainedProtocol | OnDeviceModel, ...]:
    """
    Similar to demo_model_from_cli_args, but for component models.

    Args:
    - model_cls: Must have the same length as components
    """
    res = []
    component_classes = model_cls.component_classes
    if cli_args.hub_model_id:
        if len(cli_args.hub_model_id.split(",")) != len(component_classes):
            raise ValueError(
                f"Expected {len(component_classes)} components in "
                f"hub-model-id, but got {cli_args.hub_model_id}"
            )

    cli_args_comp = copy.deepcopy(cli_args)

    for i, (cls, comp) in enumerate(
        zip(model_cls.component_classes, model_cls.component_class_names)
    ):
        if cli_args.hub_model_id:
            cli_args_comp.hub_model_id = cli_args.hub_model_id.split(",")[i]
        res.append(demo_model_from_cli_args(cls, model_id, cli_args_comp, comp))

    return tuple(res)


def demo_model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    cli_args: argparse.Namespace,
    component: str | None = None,
) -> FromPretrainedTypeVar | OnDeviceModel:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.

    If the model is a BaseModel and an on-device demo is requested,
        the BaseModel will be wrapped in an OnDeviceModel.
    """
    is_on_device = "eval_mode" in cli_args and cli_args.eval_mode == EvalMode.ON_DEVICE
    inference_model: FromPretrainedTypeVar | OnDeviceModel
    if is_on_device and issubclass(model_cls, BaseModel):
        device: hub.Device = cli_args.hub_device
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
                model_id,
                cli_args,
                additional_kwargs,
                component,
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
    model: HubModel | type[HubModel], args_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Given a dict with many args, pull out the ones relevant
    to constructing the model's input_spec.
    """
    get_input_spec_args = inspect.signature(model._get_input_spec_for_instance)
    if isinstance(model, type):
        default_args = ["self", "args", "kwargs"]
    else:
        default_args = ["args", "kwargs"]
    if list(get_input_spec_args.parameters.keys()) == default_args:
        # Use get_input_spec args if get_input_spec_for_instance is not defined.
        get_input_spec_args = inspect.signature(model.get_input_spec)

    input_spec_kwargs = {}
    for name in get_input_spec_args.parameters:
        if name == "self" or name not in args_dict:
            continue
        input_spec_kwargs[name] = args_dict[name]
    return input_spec_kwargs


def get_model_input_spec_parser(
    model_cls: type[BaseModel], parser: QAIHMArgumentParser | None = None
) -> QAIHMArgumentParser:
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


def _get_qcom_chipsets() -> set[str]:
    try:
        return {
            attr[len("chipset:") :]
            for dev in hub.get_devices()
            for attr in dev.attributes
            if attr.startswith("chipset:qualcomm")
        }
    except (APIException, UserError, InternalError):
        return set()


def _evaluate_export_common_parser(
    model_cls: type[FromPretrainedTypeVar] | type[FromPrecompiledTypeVar],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    uses_quantize_job: bool = True,
    exporting_compiled_model: bool = False,
    num_calibration_samples: int | None = None,
) -> QAIHMArgumentParser:
    """
    Common arguments between export and evaluate scripts.
    """
    # Set handler to resolve, to allow from_pretrained and get_input_spec
    # to have the same argument names.
    parser = get_parser(allow_dupe_args=True)
    if uses_quantize_job:
        parser.add_argument(
            "--num-calibration-samples",
            type=int,
            default=num_calibration_samples,
            help="The number of calibration data samples to use for quantization.",
        )
    # Default runtime for compiled model is fixed for given model
    available_runtimes = set()
    for rts in supported_precision_runtimes.values():
        available_runtimes.update(rts)

    default_runtime = _get_default_runtime(available_runtimes)
    add_target_runtime_arg(
        parser,
        available_target_runtimes=available_runtimes,
        default=default_runtime,
        help="The runtime for which to export.",
    )
    if not exporting_compiled_model:
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

        supported_precisions = {
            precision
            for precision, rts in supported_precision_runtimes.items()
            if len(rts) > 0
        }
        non_float_precision = _get_non_float_precision(supported_precisions)
        add_precision_arg(
            parser,
            supported_precisions,
            default_if_arg_explicitly_passed=non_float_precision or Precision.float,
            default=Precision.float
            if Precision.float in supported_precisions
            else next(iter(supported_precisions)),
        )

    return parser


def export_parser(
    model_cls: type[FromPretrainedTypeVar] | type[FromPrecompiledTypeVar],
    components: Optional[list[str]] = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [TargetRuntime.TFLITE],
    },
    uses_quantize_job: bool = True,
    exporting_compiled_model: bool = False,
    default_export_device: str | None = None,
    num_calibration_samples: int | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in export scripts.

    Parameters:
        model_cls: Class of the model to be exported. Used to add additional
            args for model instantiation.
        components: Only used for model with component and sub-component, such
            as Llama 2, 3, where two subcomponents (e.g.,
            PromptProcessor_1, TokenGenerator_1)
            are classified under one component (e.g. Llama2_Part1_Quantized).
        supported_precision_runtimes:
            The list of supported (precision, runtime) pairs for this model.
        uses_quantize_job:
            Whether this model uses quantize job to quantize the model.
        exporting_compiled_model:
            True when exporting compiled model.
            If set, removing skip_profiling flag from export arguments.
            Default = False.
        default_export_device: Default device to set for export.
        num_calibration_samples:
            How many samples to calibrate on when quantizing by default.
            If not set, defers to the dataset to decide the number.

    Returns:
        argparse ArgumentParser object.
    """
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        exporting_compiled_model=exporting_compiled_model,
        num_calibration_samples=num_calibration_samples,
    )

    _add_device_args(parser, default_device=default_export_device)

    if uses_quantize_job:
        parser.add_argument(
            "--skip-compiling",
            action="store_true",
            help="If set, skips compiling to asset that can run on device.",
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
    parser.add_argument(
        "--fetch-static-assets",
        action="store_true",
        default=False,
        help="If true, static assets are fetched from Hugging Face, rather than re-compiling / quantizing / profiling from PyTorch.",
    )
    if components is not None or issubclass(model_cls, CollectionModel):
        choices = (
            components
            if components is not None
            else model_cls.component_class_names  # type: ignore
        )
        parser.add_argument(
            "--components",
            nargs="+",
            type=str,
            default=None,
            choices=choices,
            help="Which components of the model to be exported.",
        )
    return parser


def evaluate_parser(
    model_cls: type[FromPretrainedTypeVar] | type[FromPrecompiledTypeVar],
    supported_datasets: list[str],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [TargetRuntime.TFLITE],
    },
    uses_quantize_job: bool = True,
    num_calibration_samples: int | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in evaluate scripts.

    Parameters:
        model_cls:
            Class of the model to be exported. Used to add additional args for model instantiation.
        supported_datasets:
            List of supported dataset names.
        supported_precision_runtimes:
            The list of supported (precision, runtime) pairs for this model.
        uses_quantize_job:
            Whether this model uses quantize job to quantize the model.
        num_calibration_samples:
            How many samples to calibrate on when quantizing by default.
            If not set, defers to the dataset to decide the number.

    Returns:
        Arg parser object.
    """
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        uses_quantize_job=uses_quantize_job,
        num_calibration_samples=num_calibration_samples,
    )
    _add_device_args(parser, default_chipset="qualcomm-snapdragon-8gen3")
    if len(supported_datasets) == 0:
        return parser
    parser.add_argument(
        "--samples-per-job",
        type=int,
        default=None,
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
        default=None,
        help="Number of samples to run. If set to -1, will run on full dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use when shuffling the data. "
        "If not set, samples data deterministically.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="A compiled hub model id.",
    )
    parser.add_argument(
        "--use-dataset-cache",
        action="store_true",
        help="If set, will store hub dataset ids in a local file and re-use "
        "for subsequent evaluations on the same dataset.",
    )
    if uses_quantize_job:
        parser.add_argument(
            "--compute-quant-cpu-accuracy",
            action="store_true",
            help="If flag is set, computes the accuracy "
            "of the quantized onnx model on the CPU.",
        )
    parser.add_argument(
        "--skip-device-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy on device.",
    )
    parser.add_argument(
        "--skip-torch-accuracy",
        action="store_true",
        help="If flag is set, skips computing accuracy with the torch model.",
    )
    return parser


def enable_model_caching(parser):
    parser.add_argument(
        "--model-cache-mode",
        type=str,
        default=CacheMode.ENABLE,
        action=partial(ParseEnumAction, enum_type=CacheMode),
        choices=[cm.name.lower() for cm in list(CacheMode.__members__.values())],
        help="Cache uploaded AI Hub model during export."
        " If enable, caches uploaded model i.e. re-uses uploaded AI Hub model from cache. "
        " If disable, disables caching i.e. no reading from and write to cache."
        " If overwrite, ignores and overwrites previous cache with newly uploaded AI Hub model instead.",
    )
    return parser


def validate_precision_runtime(
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    precision: Precision,
    runtime: TargetRuntime,
):
    if (
        precision not in supported_precision_runtimes
        or runtime not in supported_precision_runtimes[precision]
    ):
        raise ValueError(
            f"Model does not support runtime {runtime} with precision {precision}. These combinations are supported: {supported_precision_runtimes}"
        )
