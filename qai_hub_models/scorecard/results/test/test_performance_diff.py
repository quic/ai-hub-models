# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff

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


def validate_perf_diff_is_empty(perf_diff):
    # No difference captured
    for _, val in perf_diff.progressions.items():
        assert len(val) == 0
    for _, val in perf_diff.regressions.items():
        assert len(val) == 0
    # No new reports captured
    assert len(perf_diff.new_perf_report) == 0
    # No missing devices found in updated report
    assert len(perf_diff.missing_devices) == 0


def test_ios_excluded():
    # Set os_name to iOS to ensure it's not included in summary
    prev_perf_metrics = get_basic_speedup_report(os_name="iOS")
    new_perf_metrics = get_basic_speedup_report(
        os_name="iOS",
        onnx_tf_inference_time=10.0,
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    # Ensure no change in perf summary
    validate_perf_diff_is_empty(perf_diff)


def test_model_inference_run_toggle():
    # Test model inference fail/pass toggle is captured
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time="null", onnx_ort_qnn_inference_time=10.0
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time="null"
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions["inf"] == [
        (MODEL_ID, "torchscript_onnx_tflite", "inf", 10.0, "null", "null", CHIPSET, OS)
    ]


def test_perf_progression_basic():
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.5, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    expected_inf_bucket = [
        (MODEL_ID, "torchscript_onnx_tflite", 20.0, 0.5, 10.0, "null", CHIPSET, OS),
    ]

    assert perf_diff.progressions[10] == expected_inf_bucket


def test_perf_regression_basic():
    # Test regression in perf numbers
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    expected_inf_bucket = [
        (MODEL_ID, "torchscript_onnx_tflite", 2, 20.0, 10.0, "null", CHIPSET, OS),
    ]

    assert perf_diff.regressions[2] == expected_inf_bucket


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

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_diff.missing_devices) == 1
    assert perf_diff.missing_devices[0] == (MODEL_ID, CHIPSET)


def test_empty_report():
    prev_perf_metrics = get_basic_speedup_report()
    prev_perf_metrics["models"][0]["performance_metrics"][0][
        "reference_device_info"
    ] = {}
    new_perf_metrics = prev_perf_metrics

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_diff.empty_perf_report) == 1
    assert perf_diff.empty_perf_report[0] == (MODEL_ID,)
