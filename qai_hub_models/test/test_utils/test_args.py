# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
import types
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.resnet18 import MODEL_ID as RESNET_MODEL_ID
from qai_hub_models.models.resnet18 import Model as ResnetModel
from qai_hub_models.models.resnet18.export import export_model as resnet_export
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    evaluate_parser,
    export_parser,
    get_export_model_name,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.export_result import ExportResult
from qai_hub_models.utils.inference import OnDeviceModel, compile_model_from_args
from qai_hub_models.utils.model_cache import CacheMode


class DynamicMockModule(types.ModuleType):
    """Create a mock module that returns a mock object for all attributes."""

    def __getattr__(self, name: str) -> MagicMock:
        return MagicMock()


@pytest.fixture(autouse=True)
def mock_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock imports of custom packages needed for some models."""
    # monkeypatch.setattr("qai_hub_models.utils.quantization_aimet_onnx", "ensure_min_aimet_onnx_version", MagicMock(return_value=True))
    for module_name in [
        "sounddevice",
        "transformers",
        "transformers.cache_utils",
        "transformers.models",
        "transformers.models.llama",
        "transformers.models.llama.modeling_llama",
        "transformers.models.whisper",
        "transformers.models.whisper.modeling_whisper",
        "transformers.modeling_attn_mask_utils",
        "matplotlib",
        "matplotlib.pyplot",
    ]:
        monkeypatch.setitem(sys.modules, module_name, DynamicMockModule(module_name))


def test_parse_resnet18_export() -> None:
    parser = export_parser(
        model_cls=ResnetModel,
        export_fn=resnet_export,
    )
    args = parser.parse_args([])
    gt_set = {
        "num_calibration_samples",
        "target_runtime",
        "compile_options",
        "profile_options",
        "quantize_options",
        "weights",
        "batch_size",
        "precision",
        "device",
        "chipset",
        "device_str",
        "device_os",
        "skip_compiling",
        "skip_profiling",
        "skip_inferencing",
        "skip_downloading",
        "skip_summary",
        "output_dir",
        "fetch_static_assets",
        "zip_assets",
    }
    assert set(vars(args).keys()) == gt_set
    assert args.target_runtime == TargetRuntime.TFLITE
    assert args.precision == Precision.float

    # Add quantized precision
    parser = export_parser(
        model_cls=ResnetModel,
        export_fn=resnet_export,
        supported_precision_runtimes={
            Precision.float: [TargetRuntime.TFLITE],
            Precision.w8a8: [
                TargetRuntime.TFLITE,
            ],
        },
    )
    args = parser.parse_args([])
    gt_set.add("quantize")
    assert set(vars(args).keys()) == gt_set
    assert args.device is None


@pytest.fixture
def llama_parser() -> argparse.ArgumentParser:
    with (
        patch(
            "qai_hub_models.utils.quantization_aimet_onnx.ensure_min_aimet_onnx_version",
            return_value=True,
        ),
        patch(
            "qai_hub_models.utils.version_helpers.ensure_supported_version",
            return_value=True,
        ),
    ):
        from qai_hub_models.models._shared.llm.export import get_llm_parser
        from qai_hub_models.models.llama_v3_1_8b_instruct import Model as LlamaModel
    return get_llm_parser(
        model_cls=LlamaModel,
        supported_precision_runtimes={
            Precision.w4a16: [
                TargetRuntime.QNN_CONTEXT_BINARY,
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                TargetRuntime.ONNXRUNTIME_GENAI,
                TargetRuntime.GENIE,
            ]
        },
        default_precision=Precision.w4a16,
        default_export_device="Snapdragon 8 Elite QRD",
    )


def test_device_parsing(llama_parser: argparse.ArgumentParser) -> None:
    device = llama_parser.parse_args(["--device", "Samsung Galaxy S25"]).device
    assert device.name == "Samsung Galaxy S25"
    assert device.attributes == []

    device = llama_parser.parse_args(["--chipset", "qualcomm-snapdragon-8gen3"]).device
    assert device.name == ""
    assert device.attributes == "chipset:qualcomm-snapdragon-8gen3"

    device = llama_parser.parse_args(
        ["--chipset", "qualcomm-snapdragon-8gen3", "--device-os", "14"]
    ).device
    assert device.os == "14"

    device = llama_parser.parse_args([]).device
    assert device.name == "Snapdragon 8 Elite QRD"

    for action in llama_parser._actions:
        if action.dest == "device_str":
            assert (
                action.help
                == "The name of the device used to run this script. Run `qai-hub list-devices` to see the list of options. If not set, defaults to `Snapdragon 8 Elite QRD`."
            )


def test_parse_llama_export(llama_parser: argparse.ArgumentParser) -> None:
    args = llama_parser.parse_args([])
    assert set(vars(args).keys()) == {
        "target_runtime",
        "compile_options",
        "profile_options",
        "link_options",
        "checkpoint",
        "host_device",
        "fp_model",
        "_skip_quantsim_creation",
        "llm_config",
        "llm_io_type",
        "sequence_length",
        "context_length",
        "precision",
        "device",
        "chipset",
        "device_os",
        "skip_profiling",
        "skip_inferencing",
        "skip_downloading",
        "skip_summary",
        "output_dir",
        "device_str",
        "model_cache_mode",
        "synchronous",
        "quantize",
        "onnx_export_dir",
        "zip_assets",
    }
    assert args.target_runtime == TargetRuntime.GENIE

    args = llama_parser.parse_args(["--do-inferencing"])
    assert args.skip_inferencing is False

    args = llama_parser.parse_args(["--do-inferencing", "--skip-inferencing"])
    assert args.skip_inferencing is True

    args = llama_parser.parse_args(["--skip-inferencing", "--do-inferencing"])
    assert args.skip_inferencing is False


def test_llama_parser_help(llama_parser: argparse.ArgumentParser) -> None:
    for action in llama_parser._actions:
        if action.option_strings[0] == "--do-inferencing":
            assert action.default is True
            assert isinstance(action, argparse._StoreFalseAction)
            assert action.dest == "skip_inferencing"
            assert (
                action.help
                == "If set, does computing on-device outputs from sample data."
            )
        if action.dest == "skip_profiling":
            assert action.default is False
            assert isinstance(action, argparse._StoreTrueAction)
            assert action.option_strings[0] == "--skip-profiling"
            assert (
                action.help
                == "If set, skips profiling of compiled model on real devices."
            )
        if action.dest == "model_cache_mode":
            assert action.default == CacheMode.DISABLE
            assert set(action.choices or []) == {"enable", "disable", "overwrite"}
            assert (
                llama_parser.parse_args(
                    ["--model-cache-mode", "overwrite"]
                ).model_cache_mode
                == CacheMode.OVERWRITE
            )


def test_parse_whisper_export() -> None:
    from qai_hub_models.models.whisper_base import Model as WhisperModel
    from qai_hub_models.models.whisper_base.export import export_model as whisper_export

    parser = export_parser(model_cls=WhisperModel, export_fn=whisper_export)
    args = parser.parse_args([])
    gt_set = {
        "num_calibration_samples",
        "target_runtime",
        "compile_options",
        "profile_options",
        "quantize_options",
        "precision",
        "device",
        "device_str",
        "chipset",
        "device_os",
        "skip_compiling",
        "skip_profiling",
        "skip_inferencing",
        "skip_downloading",
        "skip_summary",
        "output_dir",
        "fetch_static_assets",
        "components",
        "zip_assets",
    }
    assert set(vars(args).keys()) == gt_set


def test_parse_baichuan_export() -> None:
    from qai_hub_models.models.baichuan2_7b import Model as BaichuanModel
    from qai_hub_models.models.baichuan2_7b.export import (
        export_model as baichuan_export,
    )

    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.w4a16: [
            TargetRuntime.QNN_CONTEXT_BINARY,
        ],
    }

    parser = export_parser(
        model_cls=BaichuanModel,
        export_fn=baichuan_export,
        supported_precision_runtimes=supported_precision_runtimes,
    )
    args = parser.parse_args([])
    gt_set = {
        "target_runtime",
        "profile_options",
        "device",
        "device_str",
        "chipset",
        "device_os",
        "skip_profiling",
        "skip_downloading",
        "skip_summary",
        "output_dir",
        "fetch_static_assets",
        "components",
        "zip_assets",
    }
    assert set(vars(args).keys()) == gt_set


def test_parse_resnet18_evaluate() -> None:
    parser = evaluate_parser(
        model_cls=ResnetModel,
        supported_datasets=["imagenet"],
    )
    args = parser.parse_args([])
    gt_set = {
        "num_calibration_samples",
        "target_runtime",
        "compile_options",
        "profile_options",
        "quantize_options",
        "weights",
        "batch_size",
        "precision",
        "device",
        "chipset",
        "device_os",
        "samples_per_job",
        "dataset_name",
        "num_samples",
        "seed",
        "hub_model_id",
        "use_dataset_cache",
        "compute_quant_cpu_accuracy",
        "skip_device_accuracy",
        "skip_torch_accuracy",
        "device_str",
    }
    assert set(vars(args).keys()) == gt_set
    assert args.device is None
    assert args.dataset_name == "imagenet"


def test_parse_whisper_evaluate() -> None:
    from qai_hub_models.models.whisper_base import Model as WhisperModel

    parser = evaluate_parser(
        model_cls=WhisperModel,
        supported_datasets=[],
    )
    args = parser.parse_args([])
    gt_set = {
        "target_runtime",
        "compile_options",
        "profile_options",
        "quantize_options",
        "precision",
        "device",
        "chipset",
        "device_os",
        "device_str",
    }
    assert set(vars(args).keys()) == gt_set
    assert args.device is None


def test_get_export_name() -> None:
    from qai_hub_models.models.midas import Model as MidasModel
    from qai_hub_models.models.swin_tiny import Model as SwinModel

    midas_model_id = "midas"
    swin_model_id = "swin_tiny"
    assert (
        get_export_model_name(MidasModel, midas_model_id, Precision.float, {})
        == f"{midas_model_id}_float"
    )
    assert (
        get_export_model_name(
            MidasModel,
            midas_model_id,
            Precision.w8a8,
            {
                "weights": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
            },
        )
        == f"{midas_model_id}_w8a8_midas_v21_small_256"
    )

    assert (
        get_export_model_name(SwinModel, swin_model_id, Precision.float, {})
        == f"{swin_model_id}_float"
    )
    assert (
        get_export_model_name(
            SwinModel, swin_model_id, Precision.float, {"weights": "IMAGENET1K_V1"}
        )
        == f"{swin_model_id}_float"
    )
    assert (
        get_export_model_name(
            SwinModel, swin_model_id, Precision.float, {"weights": "IMAGENET1K_V2"}
        )
        == f"{swin_model_id}_float_IMAGENET1K_V2"
    )


def test_demo_model_from_cli_args() -> None:
    parser = get_model_cli_parser(ResnetModel)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)
    args = parser.parse_args(
        ["--eval-mode", "on-device", "--device", "Samsung Galaxy S25"]
    )
    validate_on_device_demo_args(args, RESNET_MODEL_ID)

    compile_mock = MagicMock(spec=hub.Model)
    compile_mock.producer = MagicMock(spec=hub.CompileJob)
    compile_mock.producer.options = ""
    with patch(
        "qai_hub_models.utils.args.compile_model_from_args", return_value=compile_mock
    ):
        model = demo_model_from_cli_args(ResnetModel, "dummy_model", args)
        assert isinstance(model, OnDeviceModel)

    args = parser.parse_args(["--eval-mode", "on-device"])
    with pytest.raises(ValueError, match="--device or --chipset must be specified"):
        validate_on_device_demo_args(args, RESNET_MODEL_ID)


def test_compile_model_from_args() -> None:
    parser = evaluate_parser(
        model_cls=ResnetModel,
        supported_datasets=["imagenet"],
        supported_precision_runtimes={
            Precision.float: [
                TargetRuntime.TFLITE,
            ],
            Precision.w8a8: [
                TargetRuntime.TFLITE,
            ],
        },
    )
    args = parser.parse_args(
        [
            "--chipset",
            "qualcomm-snapdragon-8gen3",
            "--compile-options",
            "'--qairt_version=2.39'",
            "--quantize-options",
            "'--range_scheme min_max'",
            "--precision",
            "w8a8",
        ]
    )
    with patch(
        "qai_hub_models.models.resnet18.export.export_model"
    ) as resnet_export_mock:
        mock_compile_job = create_autospec(hub.CompileJob)
        mock_compile_job._target_model = create_autospec(hub.Model)
        resnet_export_mock.return_value = ExportResult(compile_job=mock_compile_job)
        compile_model_from_args(RESNET_MODEL_ID, args, {})
        kwargs = resnet_export_mock.call_args_list[0][1]
        assert isinstance(kwargs["device"], hub.Device)
        assert kwargs["device"].attributes == "chipset:qualcomm-snapdragon-8gen3"
        assert kwargs["compile_options"] == "'--qairt_version=2.39'"
        assert kwargs["quantize_options"] == "'--range_scheme min_max'"


def test_model_parser_uses_docstrings_for_help() -> None:
    """Test that get_model_cli_parser uses docstrings for help messages."""
    parser = get_model_cli_parser(ResnetModel)

    # Find the --weights action and check its help message
    weights_action = None
    for action in parser._actions:
        if "--weights" in action.option_strings:
            weights_action = action
            break

    assert weights_action is not None, "Expected --weights argument"
    # The help should contain the docstring content, not the generic fallback
    assert (
        weights_action.help is not None and "Pre-trained weights" in weights_action.help
    )
    assert (
        weights_action.help is not None
        and "For documentation, see" not in weights_action.help
    )
