# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch


class InfenceMetrics(NamedTuple):
    psnr: float
    shape: Tuple[int, ...]


def torch_inference(
    model: torch.nn.Module, sample_inputs: Dict[str, List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Performs inference on a torch model given a set of sample inputs.

    Parameters:
        model: The torch model.
        sample_inputs: Map from input name to list of values for that input.

    Returns:
        List of numpy array outputs,
    """
    torch_outs: List[List[torch.Tensor]] = []
    input_names = sample_inputs.keys()
    for i in range(len(list(sample_inputs.values())[0])):
        inputs = {}
        for input_name in input_names:
            inputs[input_name] = torch.from_numpy(sample_inputs[input_name][i])
        with torch.no_grad():
            out = model(**inputs)
        out_tuple = (out,) if isinstance(out, torch.Tensor) else out
        for i, out_val in enumerate(out_tuple):
            if i == len(torch_outs):
                torch_outs.append([])
            torch_outs[i].append(out_val)
    return [torch.cat(out_list, dim=0).numpy() for out_list in torch_outs]


def compute_psnr(
    output_a: Union[torch.Tensor, np.ndarray],
    output_b: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-5,
    eps2: float = 1e-10,
) -> float:
    """
    Computes the PSNR between two tensors.
    """
    if not isinstance(output_a, np.ndarray):
        a = output_a.detach().numpy().flatten()
    else:
        a = output_a.flatten()
    if not isinstance(output_b, np.ndarray):
        b = output_b.detach().numpy().flatten()
    else:
        b = output_b.flatten()
    max_b = np.abs(b).max()
    sumdeltasq = 0.0
    sumdeltasq = ((a - b) * (a - b)).sum()
    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    return 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))


def compute_relative_error(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    assert expected.shape == actual.shape
    return (np.abs(expected - actual) / (np.abs(expected) + 1e-20)).flatten()


def compare_psnr(
    output_a: Union[torch.Tensor, np.ndarray],
    output_b: Union[torch.Tensor, np.ndarray],
    psnr_threshold: int,
    eps: float = 1e-5,
    eps2: float = 1e-10,
) -> None:
    """
    Raises an error if the PSNR between two tensors is above a threshold.
    """
    psnr = compute_psnr(output_a, output_b, eps, eps2)
    assert psnr > psnr_threshold


def generate_comparison_metrics(
    expected: List[np.ndarray], actual: List[np.ndarray]
) -> Dict[int, InfenceMetrics]:
    """
    Compares the outputs of a model run in two different ways.
    For example, expected might be run on local cpu and actual run on device.

    Parameters:
        expected: List of numpy array outputs computed from a ground truth model.
        actual: List of numpy array outputs computed from an experimental model.

    Returns:
        A set of metrics representing how close the two sets of outputs are.
    """
    metrics = {}
    for i, (expected_arr, actual_arr) in enumerate(zip(expected, actual)):
        metrics[i] = InfenceMetrics(
            compute_psnr(expected_arr, actual_arr), expected_arr.shape
        )
    return metrics
