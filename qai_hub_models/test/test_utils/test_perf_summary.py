# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

import ruamel.yaml

from qai_hub_models.utils.scorecard.perf_summary import PerformanceSummary

CHIPSET = "GEN2"
OS = "13"
MODEL_ID = "dummy"


def get_basic_speedup_report(
    os_name: str = "Android",
    onnx_tf_inference_time="null",
    onnx_ort_qnn_inference_time=100.0,
):
    return {
        "models": [
            {
                "name": "dummy",
                "performance_metrics": [
                    {
                        "reference_device_info": {
                            "os": OS,
                            "os_name": os_name,
                            "chipset": CHIPSET,
                        },
                        "torchscript_onnx_tflite": {
                            "inference_time": onnx_tf_inference_time,
                        },
                        "torchscript_onnx_qnn": {
                            "inference_time": 5.0,
                        },
                        "torchscript_qnn": {
                            "inference_time": 5.0,
                        },
                    },
                ],
            },
        ]
    }


def read_config(config_path):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.preserve_yaml_order = True
    with open(config_path, "r") as file:
        return yaml.load(file)


def validate_perf_summary_is_empty(perf_summary):
    # No difference captured
    for _, val in perf_summary.progressions.items():
        assert len(val) == 0
    for _, val in perf_summary.regressions.items():
        assert len(val) == 0
    # No new reports captured
    assert len(perf_summary.new_perf_report) == 0
    # No missing devices found in updated report
    assert len(perf_summary.missing_devices) == 0


def test_ios_excluded():
    # Set os_name to iOS to ensure it's not included in summary
    prev_perf_metrics = get_basic_speedup_report(os_name="iOS")
    new_perf_metrics = get_basic_speedup_report(
        os_name="iOS",
        onnx_tf_inference_time=10.0,
    )

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    # Ensure no change in perf summary
    validate_perf_summary_is_empty(perf_summary)


def test_model_inference_run_toggle():
    # Test model inference fail/pass toggle is captured
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time="null", onnx_ort_qnn_inference_time=10.0
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time="null"
    )

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_summary.progressions["inf"] == [
        (MODEL_ID, "torchscript_onnx_tflite", "inf", 10.0, "null", CHIPSET, OS)
    ]


def test_perf_progression_basic():
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.5, onnx_ort_qnn_inference_time=5.123
    )

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    expected_inf_bucket = [
        (MODEL_ID, "torchscript_onnx_tflite", 20.0, 0.5, 10.0, CHIPSET, OS),
    ]

    assert perf_summary.progressions[10] == expected_inf_bucket


def test_perf_regression_basic():
    # Test regression in perf numbers
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    expected_inf_bucket = [
        (MODEL_ID, "torchscript_onnx_tflite", 2, 20.0, 10.0, CHIPSET, OS),
    ]

    assert perf_summary.regressions[2] == expected_inf_bucket


def test_missing_devices():
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=1.123, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.372, onnx_ort_qnn_inference_time=5.123
    )

    # Override chipset
    new_perf_metrics["models"][0]["performance_metrics"][0]["reference_device_info"][
        "chipset"
    ] = "diff-chip-xyz"

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_summary.missing_devices) == 1
    assert perf_summary.missing_devices[0] == (MODEL_ID, CHIPSET)


def test_empty_report():
    prev_perf_metrics = get_basic_speedup_report()
    prev_perf_metrics["models"][0]["performance_metrics"][0][
        "reference_device_info"
    ] = {}
    new_perf_metrics = prev_perf_metrics

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    # Update perf summary
    perf_summary.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_summary.empty_perf_report) == 1
    assert perf_summary.empty_perf_report[0] == (MODEL_ID,)


def test_e2e_aotgan_perf_summary_no_change():
    perf_filename = os.path.join(os.path.dirname(__file__), "perf.yaml")

    # Ensure perf.yaml is present, if moved, please make accordingly changes in the script.
    assert os.path.exists(os.path.join(perf_filename))

    perf_summary = PerformanceSummary()
    validate_perf_summary_is_empty(perf_summary)

    existing_model_card = read_config(perf_filename)
    perf_summary.update_summary(
        "aotgan",
        previous_report=existing_model_card,
        new_report=existing_model_card,
    )

    # Ensure perf summary is empty
    validate_perf_summary_is_empty(perf_summary)
