# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from qai_hub_models.models._shared.llm import export
from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx
from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.utils.args import QAIHMArgumentParser
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.model_cache import CacheMode

TEST_GENAI_RUNTIMES = [x for x in TargetRuntime if x.is_exclusively_for_genai]
TEST_NUM_COMPONENTS = [1, 3]
TEST_HUB_JOB_OPTIONS = ["", f"{QAIRTVersion.HUB_FLAG} 2.36", "--extra-option"]
TEST_HUB_DEVICES: list[tuple[str | None, str | None, str, str, bool]] = [
    # Device Name, Device Chipset, DSP Arch, SOC Model, supports_weight_sharing
    ("Samsung Galaxy S24 Ultra", None, "v75", "57", True),
    (None, "snapdragon-x-elite", "v73", "60", True),
    ("RB3 Gen 2", None, "v68", "35", False),
]
EXPORT = "qai_hub_models.models._shared.llm.export"


class DummyMockModel(LLM_AIMETOnnx):
    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device | None = None,
        sequence_length: int = 128,
        context_length: int = 4096,
        precision: Precision | None = None,
        fp_model: torch.nn.Module | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
        llm_config: dict[str, Any] | None = None,
        model_mock_side_effects: dict[str, Any] | None = None,
    ):
        """
        Get a mocked instance of the LLM_AIMETOnnx class.
        Provides real return values for methods used during export without actually instantiating an LLM.

        Parameters
        ----------
            sequence_length:
                The sequence length of the LLM.
            context_length:
                The context length of the LLM.
            llm_config:
                The LLM config, or None for the default config.
            model_mock_side_effects:
                Additional (function name, side effect) pairs to mock on the model object beyond the standard mocks.
        """
        model = mock.Mock(DummyMockModel)
        model.llm_config = mock.MagicMock()
        model.llm_config.to_dict.side_effect = lambda: llm_config or {}
        model.sequence_length = sequence_length
        model.context_length = context_length
        model.get_input_spec = DummyMockModel.get_input_spec
        model.get_hub_compile_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_compile_options, model
        )
        model.get_hub_profile_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_profile_options, model
        )
        model.get_hub_link_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_link_options, model
        )
        model.get_qairt_context_graph_name.side_effect = functools.partial(
            LLM_AIMETOnnx.get_qairt_context_graph_name, model
        )

        if model_mock_side_effects is not None:
            for attrname, effect in model_mock_side_effects.items():
                getattr(model, attrname).side_effect = effect

        return model

    @staticmethod
    def get_input_spec(
        input_spec_arg_a: int = 1, input_spec_arg_b: int = 2, *args, **kwargs
    ) -> InputSpec:
        return {}


@pytest.mark.parametrize(
    (
        "target_runtime",
        "host_device",
        "checkpoint",
        "device",
        "chipset",
        "skip_profiling",
        "skip_inferencing",
        "skip_downloading",
        "skip_summary",
        "output_dir",
        "compile_options",
        "link_options",
        "profile_options",
        "synchronous",
        "model_cache_mode",
    ),
    [
        (
            TargetRuntime.ONNXRUNTIME_GENAI,
            torch.device("cpu"),
            "my/checkpoint/path",
            None,
            "qualcomm-snapdragon-x-elite",
            True,
            False,
            True,
            False,
            "my_output_dir",
            "",
            "--link-option hello",
            "",
            False,
            CacheMode.DISABLE,
        ),
        (
            None,
            None,
            "DEFAULT",
            "Snapdragon X Elite CRD",
            None,
            False,
            True,
            False,
            None,
            None,
            "--compile-option world",
            None,
            "--profile-option hello",
            True,
            CacheMode.ENABLE,
        ),
    ],
)
def test_export_cli(
    # Args users can pass in.
    target_runtime: export.VALID_TARGET_RUNTIMES | None,
    host_device: torch.device | None,
    checkpoint: str | None,
    device: str | None,
    chipset: str | None,
    skip_profiling: bool,
    skip_inferencing: bool,
    skip_downloading: bool,
    skip_summary: bool,
    output_dir: str | None,
    compile_options: str,
    link_options: str,
    profile_options: str,
    synchronous: bool,
    model_cache_mode: CacheMode,
):
    """
    Test helper for verifying cli args are:
        * parsed by the export script's main function
        * passed to the llm export_model function
    """
    model_cls = DummyMockModel
    fp_model_cls = mock.MagicMock()
    model_id = "my_dummy_model"
    model_asset_version = 2
    position_processor_cls = mock.MagicMock()
    num_splits = 5
    num_layers_per_split = 10
    default_export_device = "Snapdragon X Elite CRD"
    default_precision = Precision.w8a16
    supported_precision_runtimes = {
        Precision.w4: list(TEST_GENAI_RUNTIMES),
        Precision.w8a16: list(TEST_GENAI_RUNTIMES),
    }

    # Required args to LLM_AimetONNX.from_pretrained() that are optional args for the user
    user_provided_from_pretrained_kwargs = dict(
        sequence_length=1280,
        context_length=40960,
        host_device=host_device,
        checkpoint=checkpoint,
    )

    args: dict[str, Any] = dict(
        # Required args to export_model() that aren't passed by the user
        model_cls=model_cls,
        num_splits=num_splits,
        num_layers_per_split=num_layers_per_split,
        model_asset_version=model_asset_version,
        position_processor_cls=position_processor_cls,
        # Required args to LLM_AimetONNX.from_pretrained()
        _skip_quantsim_creation=(bool(skip_inferencing)),
        **user_provided_from_pretrained_kwargs,
    )

    cli_args = ["export.py"]
    for k, v in user_provided_from_pretrained_kwargs.items():
        if v is not None:
            cli_args.extend([f"--{k.replace('_', '-')}", str(v)])

    if target_runtime:
        args["target_runtime"] = target_runtime
        cli_args.extend(["--target-runtime", target_runtime.value])
    if checkpoint:
        args["checkpoint"] = checkpoint
        cli_args.extend(["--checkpoint", checkpoint])
    if device:
        cli_args.extend(["--device", device])
    if chipset:
        cli_args.extend(["--chipset", chipset])
    if skip_profiling:
        cli_args.append("--skip-profiling")
        args["skip_profiling"] = True
    if not skip_inferencing:
        cli_args.append("--do-inferencing")
        args["skip_inferencing"] = False
    if skip_downloading:
        cli_args.append("--skip-downloading")
        args["skip_downloading"] = True
    if skip_summary:
        cli_args.append("--skip-summary")
        args["skip_summary"] = True
    if output_dir:
        cli_args.extend(["--output-dir", output_dir])
        args["output_dir"] = output_dir
    if compile_options:
        cli_args.extend(["--compile-options", compile_options])
        args["compile_options"] = compile_options
    if link_options:
        cli_args.extend(["--link-options", link_options])
        args["link_options"] = link_options
    if profile_options:
        cli_args.extend(["--profile-options", profile_options])
        args["profile_options"] = profile_options
    if synchronous:
        cli_args.append("--synchronous")
        args["synchronous"] = True
    if model_cache_mode:
        cli_args.extend(["--model-cache-mode", model_cache_mode.value])
        args["model_cache_mode"] = model_cache_mode
    args["device"] = QAIHMArgumentParser.get_hub_device(device, chipset)

    original_export = import_module(EXPORT).export_model
    with (
        mock.patch(
            f"{EXPORT}.export_model", return_value=None, autospec=True
        ) as export_mock,
        mock.patch.object(sys, "argv", cli_args),
    ):
        export_mock.__doc__ = original_export.__doc__
        export.export_main(
            model_id,
            model_asset_version,
            supported_precision_runtimes,
            num_splits,
            num_layers_per_split,
            model_cls,
            fp_model_cls,
            position_processor_cls,
            default_export_device,
            default_precision,
        )
        export_mock.assert_called_once()
        export_kwargs = export_mock.call_args_list[0].kwargs
        for k, v in args.items():
            assert k in export_kwargs and export_kwargs[k] == v
