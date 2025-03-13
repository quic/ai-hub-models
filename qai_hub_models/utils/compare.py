# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last


def _flatten_tuple(out_tuple):
    if isinstance(out_tuple, torch.Tensor):
        return (out_tuple.detach(),)
    elif isinstance(out_tuple, Iterable):
        out_tuple = tuple(out_tuple)
    else:
        raise ValueError(
            f"Invalid type for out_tuple: {type(out_tuple)}. "
            "Expected torch.Tensor or Iterable."
        )

    flattened_tuple = []
    for elem in out_tuple:
        flattened_tuple.extend(_flatten_tuple(elem))

    return tuple(flattened_tuple)


def _torch_inference_impl(
    model: BaseModel,
    sample_inputs: dict[str, list[np.ndarray]],
) -> list[np.ndarray]:
    torch_outs: list[list[torch.Tensor]] = []
    input_names = sample_inputs.keys()
    for i in range(len(list(sample_inputs.values())[0])):
        inputs = {}
        for input_name in input_names:
            inputs[input_name] = torch.from_numpy(sample_inputs[input_name][i]).to(
                "cpu"
            )
        out = model(*inputs.values())
        out_tuple = (out,) if isinstance(out, torch.Tensor) else out
        out_tuple = _flatten_tuple(out_tuple)

        for i, out_val in enumerate(out_tuple):
            if i == len(torch_outs):
                torch_outs.append([])
            torch_outs[i].append(out_val)
    return [torch.cat(out_list, dim=0).numpy() for out_list in torch_outs]


def torch_inference(
    model: BaseModel,
    sample_inputs: dict[str, list[np.ndarray]],
    return_channel_last_output: bool = True,
) -> list[np.ndarray]:
    """
    Performs inference on a torch model given a set of sample inputs.

    Parameters:
        model: The torch model.
        sample_inputs: Map from input name to list of values for that input.
        return_channel_last_output: If set, will transpose outputs to channel last
            format. Will only transpose the outputs specified by
            `model.get_channel_last_outputs()`.

    Returns:
        List of numpy array outputs,
    """
    numpy_outputs = _torch_inference_impl(model, sample_inputs)
    if not return_channel_last_output:
        return numpy_outputs
    new_outputs = []
    channel_last_outputs = model.get_channel_last_outputs()
    for name, np_out in zip(model.get_output_names(), numpy_outputs):
        if name in channel_last_outputs:
            new_out = transpose_channel_first_to_last([name], {name: [np_out]})
            new_outputs.append(new_out[name][0])
        else:
            new_outputs.append(np_out)
    return new_outputs


def compute_psnr(
    output_a: Union[torch.Tensor, np.ndarray],
    output_b: Union[torch.Tensor, np.ndarray],
    data_range: float | None = None,
    eps: float = 1e-5,
    eps2: float = 1e-10,
) -> float:
    """
    Computes the PSNR between two tensors, where output_b is the ground
    truth used to estimate data_range if `data_range` is not provided.
    """
    if not isinstance(output_a, np.ndarray):
        a = output_a.detach().float().numpy().flatten()
    else:
        a = output_a.flatten().astype(np.float32)
    if not isinstance(output_b, np.ndarray):
        b = output_b.detach().float().numpy().flatten()
    else:
        b = output_b.flatten().astype(np.float32)

    sumdeltasq = 0.0
    sumdeltasq = ((a - b) * (a - b)).sum()
    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    data_range_est = data_range if data_range is not None else np.abs(b).max()

    return 20 * np.log10((data_range_est + eps) / (sumdeltasq + eps2))


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


def compute_top_k_accuracy(expected, actual, k):
    """
    expected, actual: logit / softmax prediction of the same 1D shape.
    """
    top_k_expected = np.argpartition(expected.flatten(), -k)[-k:]
    top_k_actual = np.argpartition(actual.flatten(), -k)[-k:]

    top_k_accuracy = np.mean(np.isin(top_k_expected, top_k_actual))

    return top_k_accuracy


TOP_K_EXPLAINER = "Match rate between the top {k} classification predictions. 1 indicates perfect match"
PSNR_EXPLAINER = (
    "Peak Signal-to-Noise Ratio (PSNR). >30 dB is typically considered good."
)

METRICS_FUNCTIONS = dict(
    psnr=(compute_psnr, PSNR_EXPLAINER),
    top1=(
        lambda expected, actual: compute_top_k_accuracy(expected, actual, 1),
        TOP_K_EXPLAINER.format(k=1),
    ),
    top5=(
        lambda expected, actual: compute_top_k_accuracy(expected, actual, 5),
        TOP_K_EXPLAINER.format(k=5),
    ),
)


def generate_comparison_metrics(
    expected: list[np.ndarray],
    actual: list[np.ndarray],
    names: Optional[list[str]] = None,
    metrics: str = "psnr",
) -> pd.DataFrame:
    """
    Compares the outputs of a model run in two different ways.
    For example, expected might be run on local cpu and actual run on device.

    Parameters:
        expected: List of numpy array outputs computed from a ground truth model.
        actual: List of numpy array outputs computed from an experimental model.
        metrics: comma-separated metrics names, e.g., "psnr,top1,top5"

    Returns:
        DataFrame with range index (0, 1, 2...) and shape,  metrics as columns
        (e.g., shape | psnr | top1 | top5.
    """
    metrics_ls = metrics.split(",")
    for m in metrics_ls:
        supported_metrics = ", ".join(METRICS_FUNCTIONS.keys())
        if m not in METRICS_FUNCTIONS.keys():
            raise ValueError(
                f"Metrics {m} not supported. Supported metrics: {supported_metrics}"
            )
    idx = (
        pd.Index(names, name="output_name")
        if names
        else pd.RangeIndex(stop=len(expected))
    )
    df_res = pd.DataFrame(None, columns=["shape"] + metrics_ls, index=idx)
    for i, (expected_arr, actual_arr) in enumerate(zip(expected, actual)):
        loc = i if not names else names[i]
        df_res.loc[
            loc, "shape"
        ] = expected_arr.shape  # pyright: ignore[reportArgumentType,reportCallIssue]
        for m in metrics_ls:
            df_res.loc[loc, m] = METRICS_FUNCTIONS[m][0](expected_arr, actual_arr)  # type: ignore[operator]
    return df_res
