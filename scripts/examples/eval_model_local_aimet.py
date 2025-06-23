# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import importlib
import json
import math
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import qai_hub as hub
import torch
from aimet_common.defs import CallbackFunc, QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_onnx import quant_analyzer
from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from qai_hub.client import CompileJob
from tqdm import tqdm

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import (
    evaluate_session_on_dataset,
    get_deterministic_sample,
)
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.onnx_helpers import mock_torch_onnx_inference
from qai_hub_models.utils.path_helpers import MODEL_IDS

QUANT_RESULTS_PATH = os.environ.get("QUANT_RESULTS_PATH", os.path.expanduser("~"))
RESULTS_FOLDER = Path(QUANT_RESULTS_PATH) / "quant_debug"
HIGHER_PRECISION_FOR_MIXED_PRECISION = "w16a16"


def _make_dummy_inputs(input_spec) -> dict[str, torch.Tensor]:
    tensors = make_torch_inputs(input_spec)
    dummy_inputs = dict()
    for index, input_name in enumerate(input_spec):
        dummy_inputs[input_name] = tensors[index].numpy()
    return dummy_inputs


def _calibration_forward_pass(session: onnxruntime.InferenceSession, dataloader):
    for i, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample
        for j, model_input in tqdm(enumerate(model_inputs), total=len(model_inputs)):
            torch_input = model_input.unsqueeze(0)
            mock_torch_onnx_inference(session, torch_input)


def _compute_snr(expected: np.array, actual: np.array):
    """
    Computes the SNR for two signals where the noise is defined as expected - actual
    """
    data_range = np.abs(expected).max()
    noise_pw = np.sum(np.power(expected - actual, 2))
    noise_pw /= actual.size
    noise = np.sqrt(noise_pw)
    noise = max(noise, 1e-10)
    return 20 * np.log10(data_range / noise)


def _collect_inputs_and_fp_outputs(model, dataloader, num_samples):
    model_bytes = model.SerializeToString()
    fp_session = onnxruntime.InferenceSession(
        model_bytes, providers=["CUDAExecutionProvider"]
    )

    fp_outputs = []
    fp_inputs = []
    inputs, _ = next(iter(dataloader))
    for index, input in enumerate(inputs):
        if index >= num_samples:
            break
        torch_input = input.unsqueeze(0)
        fp_inputs.append(torch_input)
        fp_outputs.append(mock_torch_onnx_inference(fp_session, torch_input))

    return fp_inputs, fp_outputs, fp_session


def _eval_accuracy(session, args):
    fp_inputs, fp_outputs = args
    quantized_outputs = []
    for input in fp_inputs:
        quantized_outputs.append(mock_torch_onnx_inference(session, input))
    snrs = []
    num_outputs = len(quantized_outputs[0])
    num_outputs = 1
    for idx in range(len(quantized_outputs)):
        if num_outputs == 1:
            snr_i = _compute_snr(
                fp_outputs[idx][0].numpy(),
                quantized_outputs[idx][0].numpy(),
            )
        else:
            snr_i = np.stack(
                [
                    _compute_snr(
                        fp_outputs[idx][i].numpy(), quantized_outputs[idx][i].numpy()
                    )
                    for i in range(num_outputs)
                ]
            ).mean()
        snrs.append(snr_i)

    avg_snr = sum(snrs) / len(snrs)
    return avg_snr if not math.isnan(avg_snr) else 0.0


def _create_aimet_quantsim(
    model_name: str,
    model: onnx.ModelProto,
    param_bw: int,
    activation_bw: int,
    quant_scheme: QuantScheme,
    dataloader,
):
    # Quantize
    sim = QuantizationSimModel(
        model,
        quant_scheme=quant_scheme,
        default_param_bw=param_bw,
        default_activation_bw=activation_bw,
        config_file=get_path_for_per_channel_config(),
        providers=["CUDAExecutionProvider"],
    )

    with compute_encodings(sim):
        _calibration_forward_pass(sim.session, dataloader)

    # Export to QDQ
    onnx_qdq_model = sim._to_onnx_qdq()
    onnx.save(
        onnx_qdq_model, str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx")
    )

    model_bytes = onnx_qdq_model.SerializeToString()
    qdq_session = onnxruntime.InferenceSession(
        model_bytes, providers=["CUDAExecutionProvider"]
    )

    return sim, qdq_session


def flip_layers_to_higher_precision(
    sim, sqnr_dict: dict, percent_to_flip: int = 10, higher_precision: str = "float"
):

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


def evaluate_and_save_results(session, model, num_samples, overall_results, tag):
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


def run_quant_analyzer(onnx_model, sim, eval_callback, input_spec, results_dir):

    # Check if we can use cached results
    if (
        Path(results_dir).is_dir()
        and Path(f"{results_dir}/per_layer_quant_enabled.json").is_file()
    ):
        print(f"Skipping QuantAnalyzer and using cached results from {results_dir}")
        sqnr_dict = json.load(open(f"{results_dir}/per_layer_quant_enabled.json"))

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
):
    overall_results = dict()

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
    model_calibration_data = get_dataset_from_name(
        calibration_dataset_name, DatasetSplit.VAL
    )
    samples_per_job = model_calibration_data.default_samples_per_job()
    dataloader = get_deterministic_sample(model_calibration_data, 100, samples_per_job)

    print("Converting model to ONNX")

    onnx_model_path = RESULTS_FOLDER / f"{model_name}" / f"{model_name}.onnx"
    if onnx_model_path.exists():
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

        onnx_model = compile_to_onnx_job.get_target_model().download()
        onnx.save(onnx_model, str(onnx_model_path))

    fp_inputs, fp_outputs, fp_session = _collect_inputs_and_fp_outputs(
        onnx_model, dataloader, 5
    )
    eval_callback = CallbackFunc(_eval_accuracy, (fp_inputs, fp_outputs))

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
            model_name, onnx_model, 8, 8, quant_scheme, dataloader
        )

        # Evaluate QDQ model
        evaluate_and_save_results(
            sim.session, model, num_samples, overall_results, "w8a8_accuracy"
        )

        sqnr_dict = None
        if apply_quant_analyzer or apply_mixed_precision:

            sqnr_dict = run_quant_analyzer(
                onnx_model,
                sim,
                eval_callback,
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
            onnx_qdq_model = sim._to_onnx_qdq()
            onnx.save(
                onnx_qdq_model,
                str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx"),
            )

            # Evaluate QDQ model
            evaluate_and_save_results(
                sim.session, model, num_samples, overall_results, "w8a8_mixed_accuracy"
            )

    # -----------
    # w8a16
    # -----------
    if eval_w816:
        onnx_model = onnx.load(str(onnx_model_path))
        sim, qdq_session = _create_aimet_quantsim(
            model_name, onnx_model, 8, 16, quant_scheme, dataloader
        )

        # Evaluate QDQ model
        evaluate_and_save_results(
            sim.session, model, num_samples, overall_results, "w8a16_accuracy"
        )

        sqnr_dict = None
        if apply_quant_analyzer or apply_mixed_precision:
            sqnr_dict = run_quant_analyzer(
                onnx_model,
                sim,
                eval_callback,
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
            onnx_qdq_model = sim._to_onnx_qdq()
            onnx.save(
                onnx_qdq_model,
                str(RESULTS_FOLDER / f"{model_name}" / f"{model_name}_qdq.onnx"),
            )

            # Evaluate QDQ model
            evaluate_and_save_results(
                sim.session, model, num_samples, overall_results, "w8a16_mixed_accuracy"
            )

    # -----------
    # w16a16
    # -----------
    if eval_w16a16:
        onnx_model = onnx.load(str(onnx_model_path))
        sim, qdq_session = _create_aimet_quantsim(
            model_name, onnx_model, 16, 16, quant_scheme, dataloader
        )

        # Evaluate QDQ model
        evaluate_and_save_results(
            sim.session, model, num_samples, overall_results, "w16a16_accuracy"
        )

    with open(RESULTS_FOLDER / f"{model_name}" / f"{model_name}.json", "w+") as f:
        try:
            results_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            results_dict = dict()

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
        help="Comma separated list of model ids (from AI Hub models). Note no spaces allowed",
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

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    OVERALL_RESULTS_FILE = str(RESULTS_FOLDER / "overall_results.json")
    models_to_analyze = args.models.split(",")

    try:
        results = json.load(open(OVERALL_RESULTS_FILE))
        print(f"Found existing results file, appending: {OVERALL_RESULTS_FILE}")
    except json.decoder.JSONDecodeError:
        print(f"Creating new results file: {OVERALL_RESULTS_FILE}")
        results = dict()

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
            )
        except Exception as e:
            print(f"Caught exception when running {model} - ", str(e))

        with open(OVERALL_RESULTS_FILE, "w+") as f:
            f.seek(0)
            json.dump(results, f, indent=4)
            f.truncate()

    print("\n")
