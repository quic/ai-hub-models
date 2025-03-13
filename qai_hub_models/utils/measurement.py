# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import qai_hub as hub
from tflite import Model as TFModel

from qai_hub_models.utils.asset_loaders import qaihm_temp_dir


def display_with_sig_figs(num: float, num_sig_figs: int = 3) -> str:
    """
    Displays the given number as a string with the appropriate number of
    significant figures. Example:
        display_with_sig_figs(1234.2, num_sig_figs=3) -> "1230"
    Parameters:
        num: Number to display.
        num_sig_figs: How many sig figs to use.
    """
    rounded_num = float(f"{num:.{num_sig_figs}g}")
    num_digits = len(str(int(rounded_num)))

    # Only display as many numbers after the decimal point to fit number of sig figs
    return f"{rounded_num:.{max(0, num_sig_figs - num_digits)}f}"


def get_formatted_size(size: float, units: list[str], unit_step_size: float) -> str:
    """
    Formats the number according to the units provided. For example:
    format_size(3600, units=["B", "KB", ...], unit_step_size=1024.0)
    would return "3.6KB"
    Parameters:
        num: Raw count of size.
        units: A list of increasing unit sizes (e.g. ["B", "KB", ...])
        unit_step_size: The ratio in size between successive units.
    """

    unit_index = 0

    while size >= unit_step_size and unit_index < len(units) - 1:
        size /= unit_step_size
        unit_index += 1

    return f"{display_with_sig_figs(size)}{units[unit_index]}"


def get_checkpoint_file_size(model_path: str, as_str: bool = True) -> Union[str, int]:
    """
    Computes how much memory the model checkpoint consumes.
    Parameters:
        model_path: Path to the model checkpoint file.
        as_str: Whether to return the result as an int or a string formatted to 2 sig figs.
    """
    num_bytes = os.path.getsize(model_path)

    if not (as_str):
        return num_bytes

    return get_formatted_size(num_bytes, [" B", " KB", " MB", " GB", " TB"], 1024.0)


def get_tflite_unique_parameters(
    model_path: str, as_str: bool = True
) -> Union[str, int]:
    """
    TFLite parameters are defined at two levels: Tensors and Buffers

    Only tensors can tell us how many parameters, but we do not want to over-count
    tensors that point to the same buffers. So, we keep track of all buffers
    we have counted through tensors.
    """
    with open(model_path, "rb") as f:
        tflite_model = f.read()
    model = TFModel.GetRootAs(tflite_model, 0)

    parameter_cnt = 0
    buffers_counted = set()
    for i in range(model.SubgraphsLength()):
        graph = model.Subgraphs(i)
        assert graph is not None
        for j in range(graph.TensorsLength()):
            tensor = graph.Tensors(j)
            assert tensor is not None
            buf_index = tensor.Buffer()

            buffer = model.Buffers(buf_index)
            assert buffer is not None
            if not buffer.DataIsNone():
                if buf_index not in buffers_counted:
                    parameter_cnt += int(np.prod(tensor.ShapeAsNumpy()))
                    buffers_counted.add(buf_index)

    if not as_str:
        return parameter_cnt

    return get_formatted_size(parameter_cnt, ["", "K", "M", "B", "T"], 1000.0)


def get_model_size_mb(hub_model: hub.Model) -> float:
    """Return target model size in MB. This is a special case for ease of
    testing"""
    assert hub_model is not None
    with qaihm_temp_dir() as tmp_dir:
        download_path = Path(tmp_dir) / "model"
        # Download the model into the temporary directory
        hub_model.download(str(download_path))
        size_mb = get_disk_size(download_path, unit="MB")
        return size_mb


def get_disk_size(path: str | Path, unit: str = "byte") -> float:
    """
    Returns file or directory size in `unit`

    Args:
    - unit: One of ["byte", "MB"]
    """
    if os.path.isdir(path):
        # Traverse the directory and add up the file sizes.
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(path)

    if unit == "MB":
        return total_size / 2**20
    return total_size
