# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Dict, List, Tuple

from prettytable import PrettyTable

RUNTIMES_TO_COMPARE = ["torchscript_onnx_qnn", "torchscript_onnx_tflite"]


class PerformanceSummary:
    """
    Generates Perf Summary of two 'performance_metrics' from perf.yaml

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
        # List of new reports being added
        self.new_perf_report: List[Tuple[str]] = []

        # Device present in previous run, but missing in new
        self.missing_devices: List = []

        # Device + runtime present in previous run, but missing in new
        self.missing_runtimes: List = []

        # Perf report with no passing job
        self.empty_perf_report: List[Tuple[str]] = []

        # Perf buckets to track
        self.perf_buckets = ["inf", 10, 5, 2, 1.5, 1.3, 1.2, 1.1, 1.05, 1.03]

        # Only track PerfSummary for Android
        self.tracked_oses: List = ["Android"]

        # Map of perf_bucket -> List of tuple of progression summary entry
        self.progressions: Dict = {}

        # Map of perf_bucket -> List of tuple of regression summary entry
        self.regressions: Dict = {}

        for each in self.perf_buckets:
            self.progressions[each] = []
            self.regressions[each] = []

    def add_missing_model(self, model_id: str):
        self.new_perf_report.append((model_id,))

    def _format_speedup(self, num):
        if isinstance(num, str):
            return num
        return float(format(num, ".5f"))

    def update_summary(self, model_id: str, previous_report, new_report):
        prev_perf_metrics = {}
        new_perf_metrics = {}

        # Create chipset to perf metric
        if previous_report is not None and new_report is not None:
            for i in range(len(previous_report["models"])):
                for j in range(len(new_report["models"])):
                    if (
                        previous_report["models"][i]["name"]
                        == new_report["models"][j]["name"]
                    ):
                        for prev_metric in previous_report["models"][i][
                            "performance_metrics"
                        ]:
                            if "chipset" in prev_metric["reference_device_info"]:
                                ref_device = prev_metric["reference_device_info"][
                                    "chipset"
                                ]
                                prev_perf_metrics[ref_device] = prev_metric

                        for new_metric in new_report["models"][j][
                            "performance_metrics"
                        ]:
                            if "chipset" in new_metric["reference_device_info"]:
                                ref_device = new_metric["reference_device_info"][
                                    "chipset"
                                ]
                                new_perf_metrics[ref_device] = new_metric

            if len(prev_perf_metrics) == 0 or len(new_perf_metrics) == 0:
                self.empty_perf_report.append((model_id,))

            for device in prev_perf_metrics.keys():
                device_info = prev_perf_metrics[device]["reference_device_info"]
                if device_info["os_name"] not in self.tracked_oses:
                    continue

                # Case 3: Chipset is missing in new data
                if device not in new_perf_metrics:
                    self.missing_devices.append((model_id, device))
                    continue

                for runtime_type in RUNTIMES_TO_COMPARE:
                    prev_inference_time = prev_perf_metrics[device].get(
                        runtime_type, {}
                    )
                    prev_inference_time = prev_inference_time.get(
                        "inference_time", "null"
                    )
                    new_inference_time = new_perf_metrics[device].get(runtime_type, {})
                    new_inference_time = new_inference_time.get(
                        "inference_time", "null"
                    )
                    if new_inference_time == prev_inference_time:
                        continue

                    if new_inference_time == "null" or prev_inference_time == "null":
                        # Case 1: Model either failed to infer or had a successful run
                        summary_entry = (
                            model_id,
                            runtime_type,
                            "inf",
                            self._format_speedup(new_inference_time),
                            self._format_speedup(prev_inference_time),
                            device_info["chipset"],
                            device_info["os"],
                        )

                        if new_inference_time == "null":
                            self.regressions["inf"].append(summary_entry)
                        else:
                            self.progressions["inf"].append(summary_entry)
                        continue

                    # Case 2: Bucketize speedup difference
                    progression_speedup = float(prev_inference_time) / float(
                        new_inference_time
                    )
                    regression_speedup = float(new_inference_time) / float(
                        prev_inference_time
                    )
                    is_progression = progression_speedup >= 1
                    speedup = (
                        progression_speedup if is_progression else regression_speedup
                    )

                    for bucket in self.perf_buckets[1:]:
                        if bucket <= speedup:  # type: ignore
                            summary = (
                                model_id,
                                runtime_type,
                                self._format_speedup(speedup),
                                self._format_speedup(new_inference_time),
                                self._format_speedup(prev_inference_time),
                                device_info["chipset"],
                                device_info["os"],
                            )
                            if is_progression:
                                self.progressions[bucket].append(summary)
                            else:
                                self.regressions[bucket].append(summary)
                            break

    def _get_summary_table(self, bucket_id, get_progressions=True):
        """
        Returns Summary Table for given bucket
        Args:
            bucket_id : bucket_id from perf_buckets
        """
        table = PrettyTable(
            [
                "Model ID",
                "Runtime",
                "Kx faster" if get_progressions else "Kx slower",
                "New Inference time",
                "Prev Inference time",
                "Chipset",
                "OS",
            ]
        )
        data = self.progressions if get_progressions else self.regressions
        rows = data[bucket_id]
        rows.sort(key=lambda k: k[2])
        table.add_rows(rows)
        return table

    def _has_perf_changes(self):
        """Returns True if there are perf changes"""
        for _, val in self.progressions.items():
            if len(val) > 0:
                return True
        for _, val in self.regressions.items():
            if len(val) > 0:
                return True
        return False

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

            if len(self.missing_devices) > 0:
                # 3. Missing devices (New runs missing data for certain devices)
                sf.write("\n----------------- Missing devices -----------------\n")
                table = PrettyTable(["Model ID", "Missing Device"])
                table.add_rows(self.missing_devices)
                sf.write(str(table))

            if len(self.new_perf_report) > 0:
                # 4. New Models (Models that did not have perf.yaml previously)
                sf.write("\n----------------- New models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows(self.new_perf_report)
                sf.write(str(table))

            if len(self.empty_perf_report) > 0:
                # 5. Empty reports (Models with no passing jobs)
                sf.write(
                    "\n----------------- Empty reports (No passing jobs) -----------------\n"
                )
                table = PrettyTable(["Model ID"])
                table.add_rows(self.empty_perf_report)
                sf.write(str(table))

        print(f"Perf change summary written to {summary_file_path}")
