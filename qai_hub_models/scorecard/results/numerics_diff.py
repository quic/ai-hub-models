# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from prettytable import PrettyTable

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.numerics_yaml import (
    QAIHMModelNumerics,
    ScorecardProfilePath,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice


class NumericsDiff:
    """Generates Numerics Difference between two numerics.yaml"""

    def __init__(self) -> None:
        self.missing_models: list[str] = []
        self.new_models: list[str] = []
        self.empty_models: list[str] = []

        # tuple<Model ID, Dataset Name, Metric Name>
        self.missing_metrics: list[tuple[str, str, str]] = []
        self.new_metrics: list[tuple[str, str, str]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device>
        self.missing_devices: list[tuple[str, str, str, ScorecardDevice]] = []
        self.new_devices: list[tuple[str, str, str, ScorecardDevice]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision>
        self.missing_precisions: list[
            tuple[str, str, str, ScorecardDevice, Precision]
        ] = []
        self.new_precisions: list[tuple[str, str, str, ScorecardDevice, Precision]] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path>
        self.missing_paths: list[
            tuple[str, str, str, ScorecardDevice, Precision, ScorecardProfilePath]
        ] = []
        self.new_paths: list[
            tuple[str, str, str, ScorecardDevice, Precision, ScorecardProfilePath]
        ] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path, FP Accuracy, Device Accuracy, Previous FP Accuracy, Previous Device Accuracy>
        # Progression == Accuracy is closer to target than previous. Regression == Accuracy is further from target than previous.
        self.progressions: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
            ]
        ] = []
        self.regressions: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
            ]
        ] = []

        # tuple<Model ID, Dataset Name, Metric Name, Device, Precision, Path, FP Accuracy, Current Device Accuracy, Difference, Difference Threshold, Newly Disabled>
        self.device_vs_float_greater_than_enablement_threshold: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
                bool,
            ]
        ] = []

    def merge_from(self, other: NumericsDiff) -> None:
        self.missing_models.extend(other.missing_models)
        self.new_models.extend(other.new_models)
        self.empty_models.extend(other.empty_models)
        self.missing_metrics.extend(other.missing_metrics)
        self.new_metrics.extend(other.new_metrics)
        self.missing_devices.extend(other.missing_devices)
        self.new_devices.extend(other.new_devices)
        self.missing_precisions.extend(other.missing_precisions)
        self.new_precisions.extend(other.new_precisions)
        self.missing_paths.extend(other.missing_paths)
        self.new_paths.extend(other.new_paths)
        self.progressions.extend(other.progressions)
        self.regressions.extend(other.regressions)
        self.device_vs_float_greater_than_enablement_threshold.extend(
            other.device_vs_float_greater_than_enablement_threshold
        )

    def _update_summary_for_path(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        precision: Precision,
        path: ScorecardProfilePath,
        previous_report: dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        | None,
        new_report: dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails],
    ) -> None:
        prev_metric = previous_report.get(path) if previous_report else None
        new_metric = new_report.get(path)

        if not prev_metric and not new_metric:
            return
        if prev_metric and not new_metric:
            self.missing_paths.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                )
            )
        elif not prev_metric and new_metric:
            self.new_paths.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                )
            )

        device_vs_float = (
            (new_metric.partial_metric - new_metric_details.partial_torch_metric)
            if new_metric
            else None
        )

        device_vs_float_prev = (
            (prev_metric.partial_metric - previous_metric_details.partial_torch_metric)
            if prev_metric and previous_metric_details
            else None
        )

        if (
            new_metric
            and device_vs_float
            and new_metric_details.metric_fp_vs_device_enablement_threshold
            and (
                abs(device_vs_float)
                >= new_metric_details.metric_fp_vs_device_enablement_threshold
            )
        ):
            info = QAIHMModelCodeGen.from_model(model_id)
            reasons = info.disabled_paths.get_disable_reasons(precision, path.runtime)
            self.device_vs_float_greater_than_enablement_threshold.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                    path,
                    f"{new_metric_details.partial_torch_metric} {new_metric_details.metric_unit}",
                    f"{new_metric.partial_metric} {new_metric_details.metric_unit}",
                    f"{device_vs_float:.3f} {new_metric_details.metric_unit}",
                    f"{new_metric_details.metric_fp_vs_device_enablement_threshold} {new_metric_details.metric_unit}",
                    reasons.scorecard_accuracy_failure is None,
                )
            )

        if (
            new_metric
            and prev_metric
            and previous_metric_details
            and new_metric_details.metric_fp_vs_device_enablement_threshold
            and device_vs_float
            and device_vs_float_prev
            and new_metric_details.metric_range.min
            == previous_metric_details.metric_range.min
            and new_metric_details.metric_range.max
            == previous_metric_details.metric_range.max
        ):
            # If both metrics are available and the range has not changed, we can compare the
            # difference between FP and Device Accuracy to find progressions and regressions.
            metric_diff_diff = abs(device_vs_float) - abs(device_vs_float_prev)
            if (
                abs(metric_diff_diff)
                > new_metric_details.metric_fp_vs_device_enablement_threshold / 10
            ):
                proreg = self.regressions if metric_diff_diff > 0 else self.progressions
                proreg.append(
                    (
                        model_id,
                        new_metric_details.dataset_name,
                        new_metric_details.metric_name,
                        device,
                        precision,
                        path,
                        f"{new_metric_details.partial_torch_metric} {new_metric_details.metric_unit}",
                        f"{new_metric.partial_metric} {new_metric_details.metric_unit}",
                        f"{previous_metric_details.partial_torch_metric} {previous_metric_details.metric_unit}",
                        f"{prev_metric.partial_metric} {previous_metric_details.metric_unit}",
                    )
                )

    def _update_summary_for_precision(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        precision: Precision,
        previous_report: dict[
            Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        ]
        | None,
        new_report: dict[
            Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
        ],
    ) -> None:
        previous_report_precision = (
            previous_report.get(precision) if previous_report else None
        )
        new_report_precision = new_report.get(precision)

        if not previous_report_precision and not new_report_precision:
            return
        if previous_report_precision and not new_report_precision:
            self.missing_precisions.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                )
            )
        elif not previous_report_precision and new_report_precision:
            self.new_precisions.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                    precision,
                )
            )

        if new_report_precision:
            all_paths: set[ScorecardProfilePath] = set()
            if previous_report_precision:
                all_paths.update(previous_report_precision.keys())
            all_paths.update(new_report_precision.keys())

            for path in all_paths:
                self._update_summary_for_path(
                    model_id,
                    previous_metric_details,
                    new_metric_details,
                    device,
                    precision,
                    path,
                    previous_report_precision,
                    new_report_precision,
                )

    def _update_summary_for_device(
        self,
        model_id: str,
        previous_metric_details: QAIHMModelNumerics.MetricDetails | None,
        new_metric_details: QAIHMModelNumerics.MetricDetails,
        device: ScorecardDevice,
        previous_report: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ]
        | None,
        new_report: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ],
    ) -> None:
        previous_report_device = (
            previous_report.get(device) if previous_report else None
        )
        new_report_device = new_report.get(device)

        if not previous_report_device and not new_report_device:
            return
        if previous_report_device and not new_report_device:
            self.missing_devices.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                )
            )
        elif not previous_report_device and new_report_device:
            self.new_devices.append(
                (
                    model_id,
                    new_metric_details.dataset_name,
                    new_metric_details.metric_name,
                    device,
                )
            )

        if new_report_device:
            all_precisions: set[Precision] = set()
            if previous_report_device:
                all_precisions.update(previous_report_device.keys())
            all_precisions.update(new_report_device.keys())

            for precision in all_precisions:
                self._update_summary_for_precision(
                    model_id,
                    previous_metric_details,
                    new_metric_details,
                    device,
                    precision,
                    previous_report_device,
                    new_report_device,
                )

    def _update_summary_for_metric(
        self,
        model_id: str,
        dataset: str,
        metric: str,
        previous_report: list[QAIHMModelNumerics.MetricDetails] | None,
        new_report: list[QAIHMModelNumerics.MetricDetails],
    ) -> None:
        previous_report_metric: QAIHMModelNumerics.MetricDetails | None = None
        new_report_metric: QAIHMModelNumerics.MetricDetails | None = None

        for metric_details in previous_report or []:
            if (
                metric_details.dataset_name == dataset
                and metric_details.metric_name == metric
            ):
                previous_report_metric = metric_details
                break

        for metric_details in new_report:
            if (
                metric_details.dataset_name == dataset
                and metric_details.metric_name == metric
            ):
                new_report_metric = metric_details
                break

        if not previous_report_metric and not new_report_metric:
            return
        if new_report_metric and not previous_report_metric:
            self.new_metrics.append((model_id, dataset, metric))
        elif previous_report_metric and not new_report_metric:
            self.missing_metrics.append((model_id, dataset, metric))

        if new_report_metric:
            all_devices: set[ScorecardDevice] = set()

            previous_report_metric_devices = None
            if previous_report_metric:
                previous_report_metric_devices = previous_report_metric.device_metric
                all_devices.update(previous_report_metric_devices.keys())

            new_report_metric_devices = new_report_metric.device_metric
            all_devices.update(new_report_metric_devices.keys())
            for device in all_devices:
                self._update_summary_for_device(
                    model_id,
                    previous_report_metric,
                    new_report_metric,
                    device,
                    previous_report_metric_devices,
                    new_report_metric_devices,
                )

    def update_summary(
        self,
        model_id: str,
        previous_report: QAIHMModelNumerics | None,
        new_report: QAIHMModelNumerics | None,
    ) -> None:
        if not new_report and not previous_report:
            return
        if new_report and not previous_report:
            self.new_models.append(model_id)
        elif previous_report and not new_report:
            self.missing_models.append(model_id)

        if new_report:
            all_metrics: set[tuple[str, str]] = set()
            all_metrics.update(
                (metric.dataset_name, metric.metric_name)
                for metric in (previous_report.metrics if previous_report else [])
            )
            all_metrics.update(
                (metric.dataset_name, metric.metric_name)
                for metric in new_report.metrics
            )
            for dataset_name, metric_name in all_metrics:
                self._update_summary_for_metric(
                    model_id,
                    dataset_name,
                    metric_name,
                    previous_report.metrics if previous_report else None,
                    new_report.metrics,
                )

    def _get_summary_table_proreg(
        self,
        data: list[
            tuple[
                str,
                str,
                str,
                ScorecardDevice,
                Precision,
                ScorecardProfilePath,
                str,
                str,
                str,
                str,
            ]
        ],
    ) -> PrettyTable:
        """
        Returns Summary Table for given bucket.

        Parameters
        ----------
        data
            List of tuples containing model id, dataset name, metric name, device,
            precision, path, FP accuracy, device accuracy, previous FP accuracy,
            and previous device accuracy.

        Returns
        -------
        table
            Summary table for the given data.
        """
        table = PrettyTable(
            [
                "Model ID",
                "Dataset Name",
                "Metric Name",
                "Device",
                "Precision",
                "Runtime",
                "FP Accuracy",
                "Device Accuracy",
                "Previous FP Accuracy",
                "Previous Device Accuracy",
            ]
        )
        data.sort(key=lambda k: k[0])  # sort by model id
        table.add_rows(data)
        return table

    def dump_summary(self, summary_file_path: str) -> None:
        """Dumps Perf change summary captured so far to the provided path."""
        with open(summary_file_path, "w") as sf:
            sf.write("================= Perf Change Summary =================")
            sf.write(
                "\n\n----------------- Disabled Configurations -----------------\n"
            )
            if self.device_vs_float_greater_than_enablement_threshold:
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Bane",
                        "Metric name",
                        "Device",
                        "Precision",
                        "Runtime",
                        "FP Accuracy",
                        "Device Accuracy",
                        "Difference",
                        "Difference Threshold",
                        "Newly Disabled",
                    ]
                )
                self.device_vs_float_greater_than_enablement_threshold.sort(
                    key=lambda k: k[0]
                )  # sort by model id
                table.add_rows(self.device_vs_float_greater_than_enablement_threshold)
                sf.write(str(table))
            else:
                sf.write("No disabled configurations found.\n")
            sf.write("\n")

            sf.write("\n\n----------------- Regressions -----------------\n")
            if self.regressions:
                sf.write(str(object=self._get_summary_table_proreg(self.regressions)))
            else:
                sf.write("No significant regressions observed.\n")
            sf.write("\n")

            sf.write("\n\n----------------- Progressions -----------------\n")
            if self.progressions:
                sf.write(str(object=self._get_summary_table_proreg(self.progressions)))
            else:
                sf.write("No significant progressions observed.\n")
            sf.write("\n")

            if self.missing_models:
                sf.write("\n----------------- Missing Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.missing_models)
                sf.write(str(table))
                sf.write("\n")

            if self.new_models:
                sf.write("\n----------------- New Models -----------------\n")
                table = PrettyTable(["Model ID"])
                table.add_rows([x] for x in self.new_models)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_metrics:
                sf.write("\n----------------- Missing Metrics -----------------\n")
                table = PrettyTable(["Model ID", "Dataset Name", "Metric Name"])
                table.add_rows(self.missing_metrics)
                sf.write(str(table))
                sf.write("\n")

            if self.new_metrics:
                sf.write("\n----------------- New Metrics -----------------\n")
                table = PrettyTable(["Model ID", "Dataset Name", "Metric Name"])
                table.add_rows(self.new_metrics)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_devices:
                sf.write("\n----------------- Missing Devices -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device"]
                )
                table.add_rows(self.missing_devices)
                sf.write(str(table))
                sf.write("\n")

            if self.new_devices:
                sf.write("\n----------------- New Devices -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device"]
                )
                table.add_rows(self.new_devices)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_precisions:
                sf.write("\n----------------- Missing Precisions -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device", "Precision"]
                )
                table.add_rows(self.missing_precisions)
                sf.write(str(table))
                sf.write("\n")

            if self.new_precisions:
                sf.write("\n----------------- New Precisions -----------------\n")
                table = PrettyTable(
                    ["Model ID", "Dataset Name", "Metric Name", "Device", "Precision"]
                )
                table.add_rows(self.new_precisions)
                sf.write(str(table))
                sf.write("\n")

            if self.missing_paths:
                sf.write("\n----------------- Missing Runtimes -----------------\n")
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Name",
                        "Metric Name",
                        "Device",
                        "Precision",
                        "Runtime",
                    ]
                )
                table.add_rows(self.missing_paths)
                sf.write(str(table))
                sf.write("\n")

            if self.new_precisions:
                sf.write("\n----------------- New Runtimes -----------------\n")
                table = PrettyTable(
                    [
                        "Model ID",
                        "Dataset Name",
                        "Metric Name",
                        "Device",
                        "Precision",
                        "Runtime",
                    ]
                )
                table.add_rows(self.new_paths)
                sf.write(str(table))
                sf.write("\n")

        print(f"Perf change summary written to {summary_file_path}")
