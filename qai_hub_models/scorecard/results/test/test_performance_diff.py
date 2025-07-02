# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import cs_8_gen_3
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff

MODEL_ID = "dummy"
JOB_ID = "jp4kr0kvg"
COMPONENT_ID = "dummy_component"


def get_basic_speedup_report(
    onnx_tf_inference_time: float | None = None,
    onnx_ort_qnn_inference_time: float | None = 100.0,
) -> QAIHMModelPerf:
    return QAIHMModelPerf(
        precisions={
            Precision.float: QAIHMModelPerf.PrecisionDetails(
                components={
                    COMPONENT_ID: QAIHMModelPerf.ComponentDetails(
                        performance_metrics={
                            cs_8_gen_3: {
                                ScorecardProfilePath.TFLITE: QAIHMModelPerf.PerformanceDetails(
                                    job_id=JOB_ID,
                                    inference_time_milliseconds=onnx_tf_inference_time,
                                ),
                                ScorecardProfilePath.ONNX: QAIHMModelPerf.PerformanceDetails(
                                    job_id=JOB_ID, inference_time_milliseconds=5.0
                                ),
                                ScorecardProfilePath.QNN_DLC: QAIHMModelPerf.PerformanceDetails(
                                    job_id=JOB_ID,
                                    inference_time_milliseconds=onnx_ort_qnn_inference_time,
                                ),
                            }
                        }
                    )
                }
            )
        }
    )


def validate_perf_diff_is_empty(perf_diff: PerformanceDiff):
    # No difference captured
    for _, val in perf_diff.progressions.items():
        assert len(val) == 0
    for _, val in perf_diff.regressions.items():
        assert len(val) == 0
    # No new reports captured
    assert len(perf_diff.new_models) == 0
    # No missing devices found in updated report
    assert len(perf_diff.missing_devices) == 0


def test_model_inference_run_toggle():
    # Test model inference fail/pass toggle is captured
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=None, onnx_ort_qnn_inference_time=10.0
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=None
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions[float("inf")] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            float("-inf"),
            10.0,
            float("inf"),
            JOB_ID,
        )
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

    assert perf_diff.progressions[10] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            10.0,
            0.5,
            20.0,
            JOB_ID,
        )
    ]


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

    assert perf_diff.regressions[2] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            10.0,
            20.0,
            2.0,
            JOB_ID,
        ),
    ]


def test_missing_devices():
    prev_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=1.123, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.372, onnx_ort_qnn_inference_time=5.123
    )

    # Override chipset
    new_perf_metrics.precisions[Precision.float].components[
        COMPONENT_ID
    ].performance_metrics.pop(cs_8_gen_3)
    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_diff.missing_devices) == 1
    assert perf_diff.missing_devices[0] == (
        MODEL_ID,
        Precision.float,
        COMPONENT_ID,
        cs_8_gen_3,
    )


def test_empty_report():
    prev_perf_metrics = get_basic_speedup_report()
    new_perf_metrics = get_basic_speedup_report()
    new_perf_metrics.precisions.pop(Precision.float)

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)
    assert perf_diff.empty_models == [MODEL_ID]
