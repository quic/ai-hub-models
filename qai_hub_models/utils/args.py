# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Utility Functions for parsing input args for export and other customer facing scripts."""

from __future__ import annotations

import argparse
import copy
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from enum import Enum
from functools import partial
from pathlib import Path
from pydoc import locate
from typing import Any, TypeVar

import qai_hub as hub
from numpydoc.docscrape import FunctionDoc

from qai_hub_models._version import __version__
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
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
)


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


class QAIHMHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """
    Argparse formatter that combnines:
      * allowing raw text (eg. newlines) in help messages
      * including defaults in help messages (except for boolean args)
    """

    def _get_help_string(self, action):
        """
        Default value for booleans in CLI help can be misleading.
        This overridden function will print just the help message for boolean args
        and print help message along with the default value for all other args.
        """
        # Don't print "(default: <value>)" in the CLI help if the value is a bool
        # or something "non-truthy" (e.g. "", None, [])
        if isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ) or (not action.default):
            return action.help
        return super()._get_help_string(action)


class QAIHMArgumentParser(argparse.ArgumentParser):
    """
    An ArgumentParser that sets hub_device from the appropriate options.
    This isn't implemented as a `type` argument to `add_argument` because the
    device/chipset can be modified by device_os.
    """

    def __init__(
        self,
        supported_precision_runtimes: (
            dict[Precision, list[TargetRuntime]] | None
        ) = None,
        default_device: str | None = None,
        default_chipset: str | None = None,
        *args,
        **kwargs,
    ):
        self.supported_precision_runtimes = supported_precision_runtimes
        self.default_device = default_device
        self.default_chipset = default_chipset
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_hub_device(
        device: str | None = None, chipset: str | None = None, device_os: str = ""
    ) -> hub.Device | None:
        """
        Get a hub.Device given a device name and/or chipset name.
        If neither is specified, the function returns None.
        """
        if chipset or device:
            return hub.Device(
                name=device or "",
                attributes=f"chipset:{chipset}" if chipset else [],
                os=device_os,
            )
        return None

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        parsed.device = self.get_hub_device(
            getattr(parsed, "device_str", None),
            getattr(parsed, "chipset", None),
            getattr(parsed, "device_os", ""),
        )
        if parsed.device is None:
            parsed.device = self.get_hub_device(
                self.default_device, self.default_chipset
            )

        if getattr(parsed, "quantize", None):
            parsed.precision = parsed.quantize

        if self.supported_precision_runtimes is not None:
            self.validate_precision_runtime(self.supported_precision_runtimes, parsed)

        return parsed

    @staticmethod
    def validate_precision_runtime(
        supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
        parsed_args: argparse.Namespace,
    ):
        """
        Verifies that supported_precision_runtimes contains the precision + runtime pair chosen by the parsed argument namespace.
        If the namespace does not include both precision and runtime, then validation is skipped.
        """
        # If fetch_static_assets is set, validation of whether a specific precision / runtime pair is supported
        # is done downstream. This validation is only necessary when running the export script.
        fetch_static_assets: str | None = getattr(
            parsed_args, "fetch_static_assets", None
        )

        # If precision or target_runtime are None, they aren't args used by this parser. This validation becomes a no-op.
        precision: Precision | None = getattr(parsed_args, "precision", None)
        target_runtime: TargetRuntime | None = getattr(
            parsed_args, "target_runtime", None
        )

        if (
            fetch_static_assets is not None
            or precision is None
            or target_runtime is None
        ):
            return

        if (
            precision not in supported_precision_runtimes
            or target_runtime not in supported_precision_runtimes[precision]
        ):
            str_supported_precision_runtimes = "\n".join(
                f"    {p}: {', '.join([rt.value for rt in rts])}"
                for p, rts in supported_precision_runtimes.items()
            )
            print(
                f"Model does not support runtime {target_runtime.value} with precision {precision}. These combinations are supported:\n"
                + str_supported_precision_runtimes
            )
            sys.exit(1)


def get_parser(
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    allow_dupe_args: bool = False,
) -> QAIHMArgumentParser:
    return QAIHMArgumentParser(
        supported_precision_runtimes=supported_precision_runtimes,
        formatter_class=QAIHMHelpFormatter,
        conflict_handler="resolve" if allow_dupe_args else "error",
    )


def _add_device_args(
    parser: QAIHMArgumentParser,
    default_device: str | None = None,
    default_chipset: str | None = None,
) -> QAIHMArgumentParser:
    # This is an assertion because this is a logic error; it shouldn't be possible to get this at runtime.
    assert not (default_device and default_chipset), (
        "Only one of default_device or default_chipset may be specified."
    )

    parser.default_device = default_device
    parser.default_chipset = default_chipset
    device_group = parser.add_argument_group("Device Selection")
    device_mutex_group = device_group.add_mutually_exclusive_group()
    device_mutex_group.add_argument(
        "--device",
        dest="device_str",
        type=str,
        help="The name of the device used to run this script. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_device}`." if default_device else ""),
    )
    device_mutex_group.add_argument(
        "--chipset",
        type=str,
        help="If set, will choose a random device with this chipset. Run `qai-hub list-devices` to see the list of options."
        + (f" If not set, defaults to `{default_chipset}`." if default_chipset else ""),
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
    helpmsg: str,
    available_target_runtimes: list[TargetRuntime] | set[TargetRuntime] | None = None,
    default: TargetRuntime = TargetRuntime.TFLITE,
) -> ParserT:
    if available_target_runtimes is None:
        available_target_runtimes = set(TargetRuntime.__members__.values())
    parser.add_argument(
        "--target-runtime",
        type=str,
        action=partial(ParseEnumAction, enum_type=TargetRuntime),  # type: ignore[arg-type]
        default=default,
        metavar=f"{{{', '.join(rt.value for rt in available_target_runtimes)}}}",
        help=helpmsg,
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
        metavar=f"{{{', '.join(str(p) for p in supported_precisions)}}}",
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
    available_target_runtimes: list[TargetRuntime] | set[TargetRuntime] | None = None,
    add_output_dir: bool = False,
    default_device: str | None = None,
):
    """
    Parameters
    ----------
    - supported_eval_modes: subset of
    EvalMode.{FP,QUANTSIM,ON_DEVICE,LOCAL_DEVICE}. Default is
    [EvalMode.FP, EvalMode.ON_DEVICE]. The first value of supported_eval_modes
    will be the default.

    - supported_precisions: subset of {Precision.float, Precision.w8a8,
    Precision.w8a16}
    """
    if available_target_runtimes is None:
        available_target_runtimes = set(TargetRuntime.__members__.values())
    if not parser:
        parser = get_parser()

    # Add --eval-mode
    supported_eval_modes = supported_eval_modes or [EvalMode.FP, EvalMode.ON_DEVICE]

    # take the first allowed mode as the default
    default_mode = supported_eval_modes[0]

    mode_help_lines = ["Run the model in one of the following modes:"]
    mode_help_lines.extend(
        f"  - {m.value}: {m.description}" for m in supported_eval_modes
    )
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
        helpmsg="The runtime to demo (if `--eval-mode on-device` is specified).",
        default=default_runtime,
        available_target_runtimes=available_target_runtimes,
    )

    # TODO: This should only include supported precisions.
    default_precisions = [Precision.float, Precision.w8a8, Precision.w8a16]
    new_supported_precisions = supported_precisions or default_precisions

    add_precision_arg(
        parser,
        set(new_supported_precisions),
        next(iter(new_supported_precisions)),
        next(iter(new_supported_precisions)),
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


def add_function_parser_args(
    signature: dict[str, inspect.Parameter],
    parser: QAIHMArgumentParser,
    help_fn: Callable[[str, Any], str],
) -> None:
    """
    Given a function signature, add the inputs to the function as args to the parser.

    Parameters
    ----------
    signature
        The function signature represented as
        a dict from arg name to parameter metadata.
    parser
        The parser object to which the args are added.
    help_fn
        A function that takes the argument name and the default value
        and returns the help string for the that arg.
    """
    for name, param in signature.items():
        # Determining type from param.annotation is non-trivial (it can be a
        # strings like "bool | None").
        bool_action = None
        arg_name = f"--{name.replace('_', '-')}"
        if param.default is not None:
            type_ = type(param.default)
            if type_ is bool:
                if param.default:
                    bool_action = "store_false"
                    # If the default is true, and the arg name does not start with no_,
                    # then add the no- to the argument (as it should be passed as --no-enable-flag, not --enable-flag)
                    if name.startswith("no_"):
                        arg_name = f"--{name[3:].replace('_', '-')}"
                    elif name.startswith("skip_"):
                        arg_name = f"--do-{name[5:].replace('_', '-')}"
                    else:
                        arg_name = f"--no-{name.replace('_', '-')}"
                else:
                    bool_action = "store_true"
                    # If the default is false, and the arg name starts with no_,
                    # then remove the no- from the argument (as it should be passed as --enable-flag, not --no-enable-flag)
                    arg_name = f"--{name.replace('_', '-')}"
        elif param.annotation == "bool":
            type_ = bool
        else:
            type_ = str

        help_str = help_fn(name, param.default)
        if bool_action:
            parser.add_argument(arg_name, dest=name, action=bool_action, help=help_str)
        elif issubclass(type_, Enum):
            parser.add_argument(
                arg_name,
                type=str,
                action=partial(ParseEnumAction, enum_type=type_),  # type: ignore[arg-type]
                default=param.default,
                choices=[
                    enum.name.lower() for enum in list(type_.__members__.values())
                ],
                help=help_str,
            )
        else:
            parser.add_argument(
                arg_name,
                dest=name,
                type=type_,
                default=param.default,
                help=help_str,
            )


def get_model_cli_parser(
    cls: type[FromPretrainedTypeVar],
    parser: QAIHMArgumentParser | None = None,
    suppress_help_arguments: list | None = None,
    allow_dupe_args: bool = False,
) -> QAIHMArgumentParser:
    """
    Generate the argument parser to create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    if not parser:
        parser = get_parser(allow_dupe_args=allow_dupe_args)

    from_pretrained_sig = inspect.signature(cls.from_pretrained)

    def get_help(name: str, default_value: Any) -> str:
        # Suppress help for argument that need not be exposed for model.
        arg_name = f"--{name.replace('_', '-')}"
        if suppress_help_arguments is not None and arg_name in suppress_help_arguments:
            return argparse.SUPPRESS
        helpmsg = (
            f"For documentation, see {cls.__name__}::from_pretrained::parameter {name}."
        )
        if default_value is True:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to False."
        elif default_value is False:
            helpmsg = f"{helpmsg} Setting this flag will set parameter {name} to True."
        return helpmsg

    signature = dict(from_pretrained_sig.parameters)
    if "cls" in signature:
        signature.pop("cls")
    if "precision" in signature:
        signature.pop("precision")
    add_function_parser_args(signature, parser, get_help)
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


def get_export_model_name(
    model_cls: type[FromPretrainedTypeVar],
    model_id: str,
    precision: Precision,
    model_kwargs: Mapping[str, Any],
) -> str:
    """
    When exporting a model with custom model_kwargs, use a different name
    for the model file saved to disk. Incorporate all customized string args
    into the name.
    """
    sig = inspect.signature(model_cls.from_pretrained, eval_str=True)

    name = f"{model_id}_{precision}"
    for key, value in sig.parameters.items():
        # Check for a simple string type.
        anno = value.annotation

        if key not in model_kwargs:
            continue

        from types import UnionType

        if not (
            anno is str
            or (
                isinstance(anno, UnionType) and any(arg is str for arg in anno.__args__)
            )
        ):
            continue

        if model_kwargs[key] != sig.parameters[key].default:
            # If the weights are a url or filepath, .stem will take the final name
            # in the path, it will also trim the suffix (i.e., yolov8n.pt -> yolov8n)
            # Note: if a string arg has a '/' character that is not part of a path,
            # we will erroneously truncate everything before it, which we're ok with.
            name += f"_{Path(str(model_kwargs[key])).stem}"
    return name


def model_from_cli_args(
    model_cls: type[FromPretrainedTypeVar], cli_args: argparse.Namespace
) -> FromPretrainedTypeVar:
    """
    Create this model from an argparse namespace.
    Default behavior is to assume the CLI args have the same names as from_pretrained method args.
    """
    return model_cls.from_pretrained(**get_model_kwargs(model_cls, vars(cli_args)))


def demo_model_components_from_cli_args(
    model_cls: type[CollectionModel],
    model_id: str,
    cli_args: argparse.Namespace,
) -> tuple[FromPretrainedProtocol | OnDeviceModel, ...]:
    """
    Similar to demo_model_from_cli_args, but for component models.

    Parameters
    ----------
    - model_cls: Must have the same length as components
    """
    res = []
    component_classes = model_cls.component_classes
    if cli_args.hub_model_id and len(cli_args.hub_model_id.split(",")) != len(
        component_classes
    ):
        raise ValueError(
            f"Expected {len(component_classes)} components in hub-model-id, but got {cli_args.hub_model_id}"
        )

    cli_args_comp = copy.deepcopy(cli_args)

    for i, (cls, comp) in enumerate(
        zip(model_cls.component_classes, model_cls.component_class_names, strict=False)
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
            print(
                f"Exported asset: {model_id}"
                + (f"::{component}" if component else "")
                + "\n"
            )

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
            if type_ is None:
                # TODO(#16652): This is brittle since it requires the parameter
                # to be imported into that scope exactly, which may not be its
                # native location.
                type_ = locate(f"{model_cls.__module__}.{param.annotation}")
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


def _evaluate_export_common_parser(
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
) -> QAIHMArgumentParser:
    """Common arguments between export and evaluate scripts."""
    # Set handler to resolve, to allow from_pretrained and get_input_spec
    # to have the same argument names.
    parser = get_parser(supported_precision_runtimes, allow_dupe_args=True)
    # Default runtime for compiled model is fixed for given model
    # Python doesn't have ordered sets, so use a dictionary to preserver order
    available_runtimes: dict[TargetRuntime, None] = {}
    for rts in supported_precision_runtimes.values():
        for rt in rts:
            available_runtimes[rt] = None

    available_runtimes_list = list(available_runtimes.keys())
    default_runtime = _get_default_runtime(available_runtimes_list)
    add_target_runtime_arg(
        parser,
        available_target_runtimes=available_runtimes_list,
        default=default_runtime,
        helpmsg="The runtime for which to export.",
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
            default=(
                Precision.float
                if Precision.float in supported_precisions
                else next(iter(supported_precisions))
            ),
        )

    return parser


def add_export_function_args(export_fn: Callable, parser: QAIHMArgumentParser) -> None:
    """
    Extracts the relevant inputs to the export function and
    adds them to the parser.
    """
    signature = dict(inspect.signature(export_fn).parameters)
    for key in [
        "components",
        "precision",
        "target_runtime",
        "additional_model_kwargs",
        # LLM specific args
        "model_cls",
        "position_processor_cls",
        "model_name",
        "model_asset_version",
        "sub_components",
        "num_layers_per_split",
        "num_splits",
    ]:
        if key in signature:
            signature.pop(key)

    if "fetch_static_assets" in signature:
        signature.pop("fetch_static_assets")
        parser.add_argument(
            "--fetch-static-assets",
            nargs="?",
            const=f"v{__version__}",
            default=None,
            help="If set, known assets are fetched rather than re-computing them. Can be passed as:\n"
            "    `--fetch-static-assets`            (get current release assets)\n"
            "    `--fetch-static-assets latest`     (get latest release assets)\n"
            "    `--fetch-static-assets v<version>` (get assets for a specific version)\n",
        )

    raw_doc = inspect.getdoc(export_fn)
    assert raw_doc is not None, "Export function must have a docstring."

    export_docs = {
        param.name: "\n".join(param.desc)
        for param in FunctionDoc(export_fn)["Parameters"]
    }

    def _get_export_help(param_name: str, default_value: Any) -> str:
        description = export_docs[param_name]
        assert description is not None, f"Input `{param_name}` must have a description."
        if default_value is True:
            description = description.replace("skips", "does")
        return description

    add_function_parser_args(signature, parser, _get_export_help)


def export_parser(
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar],
    export_fn: Callable,
    components: list[str] | None = None,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    default_export_device: str | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in export scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional
        args for model instantiation.
    components
        Only used for model with component and sub-component, such
        as Llama 2, 3, where two subcomponents (e.g.,
        PromptProcessor_1, TokenGenerator_1)
        are classified under one component (e.g. Llama2_Part1_Quantized).
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    uses_quantize_job
        Whether this model uses quantize job to quantize the model.
    exporting_compiled_model
        True when exporting compiled model.
        If set, removing skip_profiling flag from export arguments.
        Default = False.
    default_export_device
        Default device to set for export.
    num_calibration_samples
        How many samples to calibrate on when quantizing by default.
        If not set, defers to the dataset to decide the number.

    Returns
    -------
    argparse ArgumentParser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {
            Precision.float: [TargetRuntime.TFLITE],
        }
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
    )
    add_export_function_args(export_fn, parser)
    _add_device_args(parser, default_device=default_export_device)
    if components is not None or issubclass(model_cls, CollectionModel):
        choices = (
            components if components is not None else model_cls.component_class_names
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
    model_cls: type[FromPretrainedTypeVar | FromPrecompiledTypeVar],
    supported_datasets: list[str],
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] | None = None,
    uses_quantize_job: bool = True,
    num_calibration_samples: int | None = None,
) -> QAIHMArgumentParser:
    """
    Arg parser to be used in evaluate scripts.

    Parameters
    ----------
    model_cls
        Class of the model to be exported. Used to add additional args for model instantiation.
    supported_datasets
        List of supported dataset names.
    supported_precision_runtimes
        The list of supported (precision, runtime) pairs for this model.
    uses_quantize_job
        Whether this model uses quantize job to quantize the model.
    num_calibration_samples
        How many samples to calibrate on when quantizing by default.
        If not set, defers to the dataset to decide the number.

    Returns
    -------
    Arg parser object.
    """
    if supported_precision_runtimes is None:
        supported_precision_runtimes = {Precision.float: [TargetRuntime.TFLITE]}
    parser = _evaluate_export_common_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
    )
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

    _add_device_args(parser)
    if len(supported_datasets) == 0:
        return parser
    if uses_quantize_job:
        parser.add_argument(
            "--num-calibration-samples",
            type=int,
            default=num_calibration_samples,
            help="The number of calibration data samples to use for quantization.",
        )
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
        help="Random seed to use when shuffling the data. If not set, samples data deterministically.",
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
            help="If flag is set, computes the accuracy of the quantized onnx model on the CPU.",
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
