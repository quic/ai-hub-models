# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import onnx
import onnxruntime
import qai_hub as hub
import torch
from aimet_common.defs import QuantScheme
from aimet_onnx import quant_analyzer
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    _apply_constraints,
    compute_encodings,
)
from aimet_onnx.utils import OrtInferenceSession, make_psnr_eval_fn
from qai_hub.client import CompileJob
from tqdm import tqdm

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils import quantization
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import (
    evaluate_session_on_dataset,
)
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx.helpers import mock_torch_onnx_inference
from qai_hub_models.utils.path_helpers import MODEL_IDS
from qai_hub_models.utils.qai_hub_helpers import download_model_in_memory

QUANT_RESULTS_PATH = os.environ.get("QUANT_RESULTS_PATH", os.path.expanduser("~"))
RESULTS_FOLDER = Path(QUANT_RESULTS_PATH) / "quant_debug"
HIGHER_PRECISION_FOR_MIXED_PRECISION = "w16a16"


def _make_dummy_inputs(
    input_spec: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, torch.Tensor]:
    tensors = make_torch_inputs(input_spec)
    dummy_inputs = {}
    for index, input_name in enumerate(input_spec):
        dummy_inputs[input_name] = tensors[index].numpy()
    return dummy_inputs


def _calibration_forward_pass(
    session: onnxruntime.InferenceSession, dataloader: list[tuple[torch.Tensor, object]]
) -> None:
    for _, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        model_inputs, _ground_truth_values, *_ = sample
        for model_input in model_inputs:
            torch_input = model_input.unsqueeze(0)
            mock_torch_onnx_inference(session, torch_input)


def _collect_inputs_and_fp_outputs(
    model: onnx.ModelProto,
    dataloader: list[tuple[torch.Tensor, object]],
    num_samples: int,
    exec_providers: list[str],
) -> tuple[list[torch.Tensor], list[object], onnxruntime.InferenceSession]:
    model_bytes = model.SerializeToString()
    fp_session = onnxruntime.InferenceSession(model_bytes, providers=exec_providers)

    fp_outputs = []
    fp_inputs = []
    inputs, _ = next(iter(dataloader))
    for index, inp in enumerate(inputs):
        if index >= num_samples:
            break
        torch_input = inp.unsqueeze(0)
        fp_inputs.append(torch_input)
        fp_outputs.append(mock_torch_onnx_inference(fp_session, torch_input))

    return fp_inputs, fp_outputs, fp_session


def _create_aimet_quantsim(
    model_name: str,
    model: onnx.ModelProto,
    param_bw: int,
    activation_bw: int,
    quant_scheme: QuantScheme,
    dataloader: list[tuple[torch.Tensor, object]],
    exec_providers: list[str],
) -> tuple[QuantizationSimModel, onnxruntime.InferenceSession]:
    # Quantize
    with _apply_constraints(True):
        sim = QuantizationSimModel(
            model,
            quant_scheme=quant_scheme,
            default_param_bw=param_bw,
            default_activation_bw=activation_bw,
            config_file="scripts/examples/default_config.json",
            providers=exec_providers,
        )

        with compute_encodings(sim):
            _calibration_forward_pass(sim.session, dataloader)

        sim._adjust_weight_scales_for_int32_bias()

        # Export to QDQ
        onnx_qdq_model = sim.to_onnx_qdq()
        onnx.save(
            onnx_qdq_model,
            str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx"),
        )

    model_bytes = onnx_qdq_model.SerializeToString()
    qdq_session = onnxruntime.InferenceSession(model_bytes, providers=exec_providers)

    return sim, qdq_session


def flip_one_layer_to_w16(sim: QuantizationSimModel, layer_name: str) -> None:
    cg_ops = sim.connected_graph.get_all_ops()
    op = cg_ops[layer_name]
    (
        input_quantizers,
        output_quantizers,
        param_quantizers,
    ) = sim.get_op_quantizers(op)
    for q in input_quantizers + output_quantizers:
        q.set_bitwidth(16)

    for _, q in param_quantizers.items():
        q.set_bitwidth(16)


def flip_layers_to_higher_precision(
    sim: QuantizationSimModel,
    sqnr_dict: dict[str, float],
    percent_to_flip: int = 10,
    higher_precision: str = "float",
) -> None:
    sqnr_list = sorted(sqnr_dict.items(), key=lambda item: item[1])
    sqnr_list = sqnr_list[: math.ceil(len(sqnr_list) * percent_to_flip / 100)]
    cg_ops = sim.connected_graph.get_all_ops()

    for layer_name, _ in sqnr_list:
        op = cg_ops[layer_name]
        (
            input_quantizers,
            output_quantizers,
            param_quantizers,
        ) = sim.get_op_quantizers(op)
        for q in input_quantizers + output_quantizers:
            if higher_precision == "w16a16":
                q.set_bitwidth(16)
            else:
                q.enabled = False

        for _, q in param_quantizers.items():
            if higher_precision == "w16a16":
                q.set_bitwidth(16)
            else:
                q.enabled = False


def to_qdq_session(
    qdq_model: onnx.ModelProto, exec_providers: list[str]
) -> OrtInferenceSession:
    return OrtInferenceSession(qdq_model, providers=exec_providers)


def evaluate_and_save_results(
    session: onnxruntime.InferenceSession | OrtInferenceSession,
    model: BaseModel,
    num_samples: int,
    overall_results: dict[str, tuple[float, str, str]],
    tag: str,
) -> None:
    results = evaluate_session_on_dataset(
        session, model, model.eval_datasets()[0], num_samples=num_samples
    )
    accuracy, formatted_accuracy = results
    print(f"{tag}: {formatted_accuracy}")
    overall_results[tag] = (
        float(accuracy),
        formatted_accuracy,
        f"samples={num_samples}",
    )


def run_quant_analyzer(
    onnx_model: onnx.ModelProto,
    sim: QuantizationSimModel,
    eval_callback: object,
    input_spec: dict[str, tuple[tuple[int, ...], str]],
    results_dir: str,
) -> dict[str, float]:
    # Check if we can use cached results
    if (
        Path(results_dir).is_dir()
        and Path(f"{results_dir}/per_layer_quant_enabled.json").is_file()
    ):
        print(f"Skipping QuantAnalyzer and using cached results from {results_dir}")
        with open(f"{results_dir}/per_layer_quant_enabled.json") as qf:
            sqnr_dict = json.load(qf)

    else:
        analyzer = quant_analyzer.QuantAnalyzer(
            onnx_model,
            _make_dummy_inputs(input_spec),
            forward_pass_callback=eval_callback,
            eval_callback=eval_callback,
        )
        sqnr_dict = analyzer.perform_per_layer_analysis_by_enabling_quantizers(
            sim,
            results_dir=results_dir,
        )
        analyzer.export_per_layer_encoding_min_max_range(
            sim,
            results_dir=results_dir,
        )

        # Save the sensitivity dict
        with open(f"{results_dir}/per_layer_quant_enabled.json", "w") as qf:
            json.dump(sqnr_dict, qf, indent=4)

    return sqnr_dict


def debug_quant_accuracy(
    model_name: str,
    eval_w8a8: bool = True,
    eval_w816: bool = True,
    eval_w16a16: bool = True,
    apply_quant_analyzer: bool = False,
    apply_mixed_precision: bool = False,
    percent_layers_to_flip: int = 0,
    num_samples: int = 200,
    onnx_qdq_eval: bool = False,
    exec_providers: list[str] | None = None,
) -> dict[str, tuple[float, str, str]]:
    overall_results: dict[str, tuple[float, str, str]] = {}
    if exec_providers is None:
        exec_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("\n=======================================")
    print(f"Processing Model: {model_name}")
    results_path = RESULTS_FOLDER / f"{model_name}"
    results_path.mkdir(parents=True, exist_ok=True)

    # Parameters
    assert model_name in MODEL_IDS
    DEVICE_TO_EVAL = hub.Device("Snapdragon X Elite CRD")
    quant_scheme = QuantScheme.post_training_tf

    # Import model
    model_module = importlib.import_module(f"qai_hub_models.models.{model_name}")
    model_cls: type[BaseModel] = model_module.Model

    # Instantiate model in pyTorch
    model = model_cls.from_pretrained()
    input_spec = model.get_input_spec()

    print("Loading dataset")

    # Verify the model has a dataset to calibrate with
    calibration_dataset_name = model.calibration_dataset_name()
    if not calibration_dataset_name:
        calibration_dataset_name = model.eval_datasets()[0]
    assert calibration_dataset_name is not None

    calibration_data = quantization.get_calibration_data(model, input_spec, 100)

    # Use the first input name from the model's input_spec instead of hardcoding
    # the key 'image'. input_spec is an ordered dict-like mapping of input
    # names to shapes/types; take the first key to select the correct calibration
    # data entry.
    input_names = list(input_spec.keys())
    if not input_names:
        raise ValueError("input_spec has no inputs")
    input_name = input_names[0]

    dataloader = [
        (torch.from_numpy(tensor), None) for tensor in calibration_data[input_name]
    ]

    print("Converting model to ONNX")

    onnx_model_path = RESULTS_FOLDER / f"{model_name}" / f"{model_name}.onnx"
    if onnx_model_path.exists():
        print(f"Found existing ONNX model at {onnx_model_path}, loading")
        onnx_model = onnx.load(str(onnx_model_path))
    else:
        # Export model to ONNX
        traced_model = torch.jit.trace(
            model.to("cpu"), make_torch_inputs(input_spec), check_trace=False
        )
        compile_to_onnx_job: CompileJob = hub.submit_compile_job(
            model=traced_model,
            input_specs=input_spec,
            device=DEVICE_TO_EVAL,
            name=model_name,
            options=model.get_hub_compile_options(TargetRuntime.ONNX, Precision.float),
        )

        target_model = compile_to_onnx_job.get_target_model()
        assert target_model is not None
        onnx_model = download_model_in_memory(target_model)
        onnx.save(onnx_model, str(onnx_model_path))

    fp_inputs, _, fp_session = _collect_inputs_and_fp_outputs(
        onnx_model, dataloader, 5, exec_providers
    )

    input_names = [inp.name for inp in fp_session.get_inputs()]
    sensitivity_check_inputs = []
    for fp_input in fp_inputs:
        sensitivity_check_inputs.append(
            {input_names[0]: fp_input.cpu().detach().numpy()}
        )

    # -----------
    # float
    # -----------
    evaluate_and_save_results(
        fp_session, model, num_samples, overall_results, "float_accuracy"
    )

    # -----------
    # w8a8
    # -----------
    if eval_w8a8:
        sim, qdq_session = _create_aimet_quantsim(
            model_name, onnx_model, 8, 8, quant_scheme, dataloader, exec_providers
        )

        # Evaluate QDQ model
        sess_to_eval = qdq_session if onnx_qdq_eval else sim.session

        evaluate_and_save_results(
            sess_to_eval, model, num_samples, overall_results, "w8a8_accuracy"
        )

        sqnr_dict = None
        if apply_quant_analyzer or apply_mixed_precision:
            psnr_fn = make_psnr_eval_fn(fp_session, sensitivity_check_inputs, None)
            sqnr_dict = run_quant_analyzer(
                onnx_model,
                sim,
                psnr_fn,
                input_spec,
                results_dir=str(
                    RESULTS_FOLDER / f"{model_name}" / f"{model_name}_w8a8_results/"
                ),
            )

        if apply_mixed_precision:
            flip_layers_to_higher_precision(
                sim,
                sqnr_dict,
                percent_layers_to_flip,
                HIGHER_PRECISION_FOR_MIXED_PRECISION,
            )

            with compute_encodings(sim):
                _calibration_forward_pass(sim.session, dataloader)

            # Export to QDQ
            onnx_qdq_model = sim.to_onnx_qdq()
            onnx.save(
                onnx_qdq_model,
                str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx"),
            )
            qdq_session = to_qdq_session(onnx_qdq_model, exec_providers)

            # Evaluate QDQ model
            sess_to_eval = qdq_session if onnx_qdq_eval else sim.session
            evaluate_and_save_results(
                sess_to_eval, model, num_samples, overall_results, "w8a8_mixed_accuracy"
            )

    # -----------
    # w8a16
    # -----------
    if eval_w816:
        onnx_model = onnx.load(str(onnx_model_path))
        sim, qdq_session = _create_aimet_quantsim(
            model_name, onnx_model, 8, 16, quant_scheme, dataloader, exec_providers
        )

        # Evaluate QDQ model
        sess_to_eval = qdq_session if onnx_qdq_eval else sim.session

        evaluate_and_save_results(
            sess_to_eval, model, num_samples, overall_results, "w8a16_accuracy"
        )

        sqnr_dict = None
        if apply_quant_analyzer or apply_mixed_precision:
            psnr_fn = make_psnr_eval_fn(fp_session, sensitivity_check_inputs, None)
            sqnr_dict = run_quant_analyzer(
                onnx_model,
                sim,
                psnr_fn,
                input_spec,
                results_dir=str(
                    RESULTS_FOLDER / f"{model_name}" / f"{model_name}_w8a16_results/"
                ),
            )

        if apply_mixed_precision:
            flip_layers_to_higher_precision(
                sim,
                sqnr_dict,
                percent_layers_to_flip,
                HIGHER_PRECISION_FOR_MIXED_PRECISION,
            )

            with compute_encodings(sim):
                _calibration_forward_pass(sim.session, dataloader)

            # Export to QDQ
            onnx_qdq_model = sim.to_onnx_qdq()
            onnx.save(
                onnx_qdq_model,
                str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx"),
            )
            qdq_session = to_qdq_session(onnx_qdq_model, exec_providers)

            # Evaluate QDQ model
            sess_to_eval = qdq_session if onnx_qdq_eval else sim.session
            evaluate_and_save_results(
                sess_to_eval,
                model,
                num_samples,
                overall_results,
                "w8a16_mixed_accuracy",
            )

    # -----------
    # w16a16
    # -----------
    if eval_w16a16:
        onnx_model = onnx.load(str(onnx_model_path))
        sim, qdq_session = _create_aimet_quantsim(
            model_name, onnx_model, 16, 16, quant_scheme, dataloader, exec_providers
        )

        # Evaluate QDQ model
        sess_to_eval = qdq_session if onnx_qdq_eval else sim.session

        evaluate_and_save_results(
            sess_to_eval, model, num_samples, overall_results, "w16a16_accuracy"
        )

    with open(RESULTS_FOLDER / f"{model_name}" / f"{model_name}.json", "w+") as f:
        try:
            results_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            results_dict = {}

        results_dict.update(overall_results)
        overall_results = results_dict
        f.seek(0)
        json.dump(results_dict, f, indent=4)
        f.truncate()

    return overall_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use AIMET QuantSim locally to debug accuracy bottlenecks for given models"
    )

    # Add command-line arguments
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma separated list of model ids (from AI Hub Workbench models). Note no spaces allowed",
    )

    parser.add_argument(
        "--eval-w8a8", action="store_true", help="Evaluate w8a8 precision"
    )
    parser.add_argument(
        "--eval-w8a16", action="store_true", help="Evaluate w8a16 precision"
    )
    parser.add_argument(
        "--eval-w16a16", action="store_true", help="Evaluate w8a16 precision"
    )
    parser.add_argument(
        "--quant-analysis",
        action="store_true",
        help="Runs AIMET QuantAnalyzer to produce plots that help with analyzing accuracy",
    )
    parser.add_argument(
        "--apply-mixed-precision",
        action="store_true",
        help="Runs a percentage of the layers in higher precision",
    )

    parser.add_argument(
        "--percent-layers-to-flip",
        type=int,
        default=10,
        help="If --apply-mixed-precision is used, this sets the percentage of layers to set to higher precision",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to use for evaluation",
    )

    parser.add_argument(
        "--onnx-qdq-eval",
        action="store_true",
        help="If set, evaluates the QDQ model instead of the QuantSim session",
    )

    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="If set, uses the CUDA execution provider for ONNX Runtime",
    )

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if not args.use_cuda:
        exec_providers = ["CPUExecutionProvider"]
    else:
        exec_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    OVERALL_RESULTS_FILE = str(RESULTS_FOLDER / "overall_results.json")
    models_to_analyze = args.models.split(",")

    try:
        with open(OVERALL_RESULTS_FILE) as resf:
            results = json.load(resf)
        print(f"Found existing results file, appending: {OVERALL_RESULTS_FILE}")
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        print(f"Creating new results file: {OVERALL_RESULTS_FILE}")
        results = {}

    for model in models_to_analyze:
        try:
            results[model] = debug_quant_accuracy(
                model,
                eval_w8a8=args.eval_w8a8,
                eval_w816=args.eval_w8a16,
                eval_w16a16=args.eval_w16a16,
                apply_quant_analyzer=args.quant_analysis,
                apply_mixed_precision=args.apply_mixed_precision,
                percent_layers_to_flip=args.percent_layers_to_flip,
                num_samples=args.num_samples,
                onnx_qdq_eval=args.onnx_qdq_eval,
                exec_providers=exec_providers,
            )
        except Exception as e:
            print(f"Caught exception when running {model} - ", str(e))

        with open(OVERALL_RESULTS_FILE, "w+") as f:
            f.seek(0)
            json.dump(results, f, indent=4)
            f.truncate()

    print("\n")
