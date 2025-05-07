# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from prettytable import PrettyTable

from qai_hub_models.configs.devices_and_chipsets_yaml import ScorecardDevice
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf, ScorecardProfilePath
from qai_hub_models.models.common import Precision

# Last 3 values in tuple are: [prev inference time, new inference time, diff, job_id]
InferenceInfo = tuple[
    str,  # Model ID
    Precision,
    str,  # Component Name (same as Model ID if model does not have multiple components)
    ScorecardDevice,
    ScorecardProfilePath,
    float,  # prev inference time (millisecs)
    float,  # new inference time (millisecs),
    float,  # inference time diff
    str,  # Profile Job ID
]


class PerformanceDiff:
    """
    Generates Performance Difference between of two 'performance_metrics' from perf.yaml

    Perf summary is generated w.r.t 'perf_buckets' to summarize difference in decreasing order
        - "INF" -> Inference failure toggled.
        - 10 -> Speedup difference >= 10 and so on ...

    Why use speedup difference?
        - Speedup is relative to baseline measured with similar constraints and changes
        - Speedup difference gives a generate sense on the how Tetra performance diverged w.r.t. baseline at that point

    What all to capture in the summary (Summary of Interest) ?
        1. Inferences that started to fail or work (Speedup = "INF")
        2. Speedup difference >= 0.1 (check models closely from higher buckets)
        3. Missing devices (new runs missing data for certain devices)
        4. New models (models with new perf.yamls)
        5. Empty perf reports (models with no passing jobs)
    """

    def __init__(self) -> None:
        # Perf buckets to track
        self.perf_buckets: list[float] = [
            float("inf"),
            10,
            5,
            2,
            1.5,
            1.3,
            1.2,
            1.1,
        ]

        self.empty_models: list[str] = []

        self.missing_models: list[str] = []
        self.new_models: list[str] = []

        self.missing_precisions: list[tuple[str, Precision]] = []
        self.new_precisions: list[tuple[str, Precision]] = []

        self.missing_components: list[tuple[str, Precision, str]] = []
        self.new_components: list[tuple[str, Precision, str]] = []

        self.missing_devices: list[tuple[str, Precision, str, ScorecardDevice]] = []
        self.new_devices: list[tuple[str, Precision, str, ScorecardDevice]] = []

        self.progressions: dict[float | str, list[InferenceInfo]] = {
            x: [] for x in self.perf_buckets
        }
        self.regressions: dict[float | str, list[InferenceInfo]] = {
            x: [] for x in self.perf_buckets
        }

    @staticmethod
    def _format_speedup(num: float | None) -> str | float:
        if not num:
            return "null"
        return float(format(num, ".5f"))

    def _update_summary_for_path(
        self,
        model_id: str,
        precision: Precision,
        component: str,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
        previous_report: dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
        new_report: dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
    ):
        prev_inference_time = previous_report.get(
            path, QAIHMModelPerf.PerformanceDetails()
        ).inference_time_milliseconds
        new_results = new_report.get(path, QAIHMModelPerf.PerformanceDetails())
        new_inference_time = new_results.inference_time_milliseconds
        if prev_inference_time and new_inference_time:
            progression_speedup = float(prev_inference_time) / float(new_inference_time)
            regression_speedup = float(new_inference_time) / float(prev_inference_time)
            is_progression = progression_speedup >= 1
            diff = progression_speedup if is_progression else regression_speedup
        elif prev_inference_time and not new_inference_time:
            is_progression = False
            diff = float("inf")
        elif not prev_inference_time and new_inference_time:
            is_progression = True
            diff = float("inf")
        else:
            # both failed, don't add this to the summary
            return

        if is_progression:
            append_to = self.progressions
        else:
            append_to = self.regressions
        bucket = None
        for i in range(len(self.perf_buckets)):
            key = self.perf_buckets[i]
            if diff >= key:
                bucket = key
                break
        if not bucket:
            # not a meaningful change
            return
        append_to[bucket].append(
            (
                model_id,
                precision,
                component,
                device,
                path,
                prev_inference_time or float("-inf"),
                new_inference_time or float("-inf"),
                diff,
                new_results.job_id or "null",
            )
        )

    def _update_summary_for_device(
        self,
        model_id: str,
        precision: Precision,
        component: str,
        device: ScorecardDevice,
        previous_report: dict[
            ScorecardDevice,
            dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
        ],
        new_report: dict[
            ScorecardDevice,
            dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
        ],
    ):
        if device not in previous_report and device not in new_report:
            return
        if device in previous_report and device not in new_report:
            self.missing_devices.append((model_id, precision, component, device))
        elif device not in previous_report and device in new_report:
            self.new_devices.append((model_id, precision, component, device))
        else:
            all_paths: set[ScorecardProfilePath] = set()
            previous_report_device = previous_report[device]
            new_report_device = new_report[device]
            all_paths.update(previous_report_device.keys())
            all_paths.update(new_report_device.keys())
            for path in all_paths:
                self._update_summary_for_path(
                    model_id,
                    precision,
                    component,
                    device,
                    path,
                    previous_report_device,
                    new_report_device,
                )

    def _update_summary_for_component(
        self,
        model_id: str,
        precision: Precision,
        component: str,
        previous_report: QAIHMModelPerf.PrecisionDetails,
        new_report: QAIHMModelPerf.PrecisionDetails,
    ):
        if (
            component not in previous_report.components
            and component not in new_report.components
        ):
            return
        elif (
            component in previous_report.components
            and component not in new_report.components
        ):
            self.missing_components.append((model_id, precision, component))
        elif (
            component not in previous_report.components
            and component in new_report.components
        ):
            self.new_components.append((model_id, precision, component))
        else:
            all_devices: set[ScorecardDevice] = set()
            previous_report_component = previous_report.components[
                component
            ].performance_metrics
            new_report_component = new_report.components[component].performance_metrics
            all_devices.update(previous_report_component.keys())
            all_devices.update(new_report_component.keys())
            for device in all_devices:
                self._update_summary_for_device(
                    model_id,
                    precision,
                    component,
                    device,
                    previous_report_component,
                    new_report_component,
                )

    def _update_summary_for_precision(
        self,
        model_id: str,
        precision: Precision,
        previous_report: QAIHMModelPerf,
        new_report: QAIHMModelPerf,
    ):
        if (
            precision in previous_report.precisions
            and precision not in new_report.precisions
        ):
            self.missing_precisions.append((model_id, precision))
        elif (
            precision not in previous_report.precisions
            and precision in new_report.precisions
        ):
            self.new_precisions.append((model_id, precision))
        else:
            all_components: set[str] = set()
            previous_report_precision = previous_report.precisions[precision]
            new_report_precision = new_report.precisions[precision]
            all_components.update(previous_report_precision.components.keys())
            all_components.update(new_report_precision.components.keys())
            for component in all_components:
                self._update_summary_for_component(
                    model_id,
                    precision,
                    component,
                    previous_report_precision,
                    new_report_precision,
                )

    def update_summary(
        self,
        model_id: str,
        previous_report: QAIHMModelPerf | None,
        new_report: QAIHMModelPerf | None,
    ):
        if not new_report and not previous_report:
            return
        elif new_report and not previous_report:
            self.new_models.append(model_id)
        elif previous_report and not new_report:
            self.missing_models.append(model_id)
        elif previous_report and new_report:
            if new_report.empty:
                self.empty_models.append(model_id)
                return

            all_precisions: set[Precision] = set()
            all_precisions.update(previous_report.precisions.keys())
            all_precisions.update(new_report.precisions.keys())
            for precision in all_precisions:
                self._update_summary_for_precision(
                    model_id, precision, previous_report, new_report
                )

    def _get_summary_table(self, bucket_id, get_progressions=True):
        """
        Returns Summary Table for given bucket
        Args:
            bucket_id : bucket_id from perf_buckets
        """
        table = PrettyTable(
            [
                "Model ID",
                "Precision",
                "Component",
                "Device",
                "Runtime",
                "Prev Inference time",
                "New Inference time",
                "Kx faster" if get_progressions else "Kx slower",
                "Job ID",
            ]
        )
        data = self.progressions if get_progressions else self.regressions
        rows = data[bucket_id]
        rows.sort(key=lambda k: k[2])  # sort by component name
        table.add_rows(rows)
        return table

    def _has_perf_changes(self):
        """Returns True if there are perf changes"""
        return (
            self.new_components
            or self.missing_components
            or self.new_devices
            or self.missing_devices
            or self.new_models
            or self.missing_models
            or self.new_precisions
            or self.missing_precisions
            or self.progressions
            or self.regressions
        )

    def dump_summary(self, summary_file_path: str):
        """
        Dumps Perf change summary captured so far to the provided path.
        """
        with open(summary_file_path, "w") as sf:
            sf.write("================= Perf Change Summary =================")
            if self._has_perf_changes():
                sf.write("\n\n----------------- Regressions -----------------\n")
                # Dumps Point 1 and 2 from Summary of Interest
                # 1. Inferences that started to fail (Speedup = "INF")
                # 2. Slower than previous run
                for bucket in self.perf_buckets:
                    if len(self.regressions[bucket]) > 0:
                        sf.write(
                            f"\n----------------- >= {bucket}x slower -----------------\n"
                        )
                        sf.write(
                            str(self._get_summary_table(bucket, get_progressions=False))
                        )

                sf.write("\n\n----------------- Progressions -----------------\n")

                # Dumps Point 1 and 2 from Summary of Interest
                # 1. Inferences that started to work (Speedup = "INF")
                # 2. Faster than previous run
                for bucket in self.perf_buckets:
                    if len(self.progressions[bucket]) > 0:
                        sf.write(
                            f"\n----------------- >= {bucket}x faster -----------------\n"
                        )
                        sf.write(str(self._get_summary_table(bucket)))
            else:
                sf.write("\nNo significant changes observed.")

            if len(self.missing_models) > 0:
                sf.write("\n----------------- Missing Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.missing_models)
                sf.write(str(table))

            if len(self.missing_precisions) > 0:
                sf.write("\n----------------- Missing Precisions -----------------\n")
                table = PrettyTable(["Model ID", "Precision"])
                table.add_rows(self.missing_precisions)
                sf.write(str(table))

            if len(self.missing_components) > 0:
                sf.write("\n----------------- Missing Components -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component"])
                table.add_rows(self.missing_components)
                sf.write(str(table))

            if len(self.missing_devices) > 0:
                sf.write("\n----------------- Missing Devices -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component", "Device"])
                table.add_rows(self.missing_devices)
                sf.write(str(table))

            if len(self.new_models) > 0:
                sf.write("\n----------------- New Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.new_models)
                sf.write(str(table))

            if len(self.new_precisions) > 0:
                sf.write("\n----------------- New Precisions -----------------\n")
                table = PrettyTable(["Model ID", "Precision"])
                table.add_rows(self.new_precisions)
                sf.write(str(table))

            if len(self.new_components) > 0:
                # 3. Missing devices (New runs missing data for certain devices)
                sf.write("\n----------------- New Components -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component"])
                table.add_rows(self.new_components)
                sf.write(str(table))

            if len(self.new_devices) > 0:
                # 3. Missing devices (New runs missing data for certain devices)
                sf.write("\n----------------- New Devices -----------------\n")
                table = PrettyTable(["Model ID", "Precision", "Component", "Device"])
                table.add_rows(self.new_devices)
                sf.write(str(table))

            if len(self.empty_models) > 0:
                # 5. Empty reports (Models with no passing jobs)
                sf.write(
                    "\n----------------- Empty reports (No passing jobs) -----------------\n"
                )
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.empty_models)
                sf.write(str(table))

        print(f"Perf change summary written to {summary_file_path}")
