# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import qai_hub as hub
from tabulate import tabulate
from torch.utils.data import DataLoader, Sampler

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.utils.asset_loaders import (
    always_answer_prompts,
    load_yaml,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import get_dataset_path
from qai_hub_models.utils.inference import (
    AsyncOnDeviceResult,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.quantization import get_calibration_data


def skip_clone_repo_check(func):
    """
    When running QAI Hub Models functions, the user sometimes needs to type "y"
    before the repo is cloned. When testing in CI, we want to skip this check.

    Add this function as a decorator to any test function that needs to bypass this.

    @skip_clone_repo_check
    def test_fn():
        ...
    """

    def wrapper(*args, **kwargs):
        with always_answer_prompts(True):
            return func(*args, **kwargs)

    return wrapper


@pytest.fixture
def skip_clone_repo_check_fixture():
    with always_answer_prompts(True):
        yield


def assert_most_same(arr1: np.ndarray, arr2: np.ndarray, diff_tol: float) -> None:
    """
    Checks whether most values in the two numpy arrays are the same.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters:
        arr1: First input image array.
        arr2: Second input image array.
        diff_tol: Float in range [0,1] representing percentage of values
            that can be different while still having the assertion pass.

    Raises:
        AssertionError if input arrays are different size,
            or too many values are different.
    """

    different_values = arr1 != arr2
    assert (
        np.mean(different_values) <= diff_tol
    ), f"More than {diff_tol * 100}% of values were different."


def assert_most_close(
    arr1: np.ndarray,
    arr2: np.ndarray,
    diff_tol: float,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Checks whether most values in the two numpy arrays are close.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters:
        arr1: First input image array.
        arr2: Second input image array.
        diff_tol: Float in range [0,1] representing percentage of values
            that can be different while still having the assertion pass.
        atol: See rtol documentation.
        rtol: Two values a, b are considered close if the following expresion is true
            `absolute(a - b) <= (atol + rtol * absolute(b))`
            Documentation copied from `np.isclose`.

    Raises:
        AssertionError if input arrays are different size,
            or too many values are not close.
    """

    not_close_values = ~np.isclose(arr1, arr2, atol=atol, rtol=rtol)
    assert (
        np.mean(not_close_values) <= diff_tol
    ), f"More than {diff_tol * 100}% of values were not close."


def mock_first_n(fn: Callable, n: int):
    """
    Return a function that returns a Mock object for the first N calls
    and calls the given fn on all subsequent calls.
    """
    call_count = 0

    def mock_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= n:
            return mock.Mock()
        return fn(*args, **kwargs)

    return mock_fn


def mock_on_device_model_call(inference_job: hub.InferenceJob) -> Callable:
    def mock_call(self, *args):
        return AsyncOnDeviceResult(
            inference_job,
            self.target_runtime,
            self.channel_last_output,
            self.output_names,
            num_retries=0,
        )

    return mock_call


def verify_io_names(model_cls: type[BaseModel]) -> None:
    """
    Performs various checks on a model's input and output names.
    Raises an AssertionError if any checks fail.

    - Checks that the inputs and outputs specified to have the channel
    last transpose are present in the list of input and output names.
    - Checks that inputs and output names don't have dashes. The QNN compiler
    converts dashes in names to underscores, so this would create mismatch between
    the target model name and the name specified in this codebase.
    """
    input_spec = model_cls.get_input_spec()
    for channel_last_input in model_cls.get_channel_last_inputs():
        assert channel_last_input in input_spec
    output_names = model_cls.get_output_names()
    for channel_last_output in model_cls.get_channel_last_outputs():
        assert channel_last_output in output_names
    for output_name in output_names:
        assert "-" not in output_name, "output name cannot contain `-`"
    for input_name in input_spec:
        assert "-" not in input_name, "input name cannot contain `-`"


def is_hub_testing_async() -> bool:
    return bool(os.environ.get("TEST_HUB_ASYNC", 0))


def should_run_skipped_models() -> bool:
    return bool(os.environ.get("RUN_ALL_SKIPPED_MODELS", 0))


def get_artifacts_dir_opt() -> Path | None:
    adir = os.environ.get("ARTIFACTS_DIR", None)
    return Path(adir) if adir else None


def get_artifacts_dir() -> Path:
    adir = get_artifacts_dir_opt()
    assert (
        adir
    ), "Attempted to access artifacts dir, but $ARTIFACTS_DIR cli variable is not set"
    return Path(adir)


def get_artifact_filepath(filename, artifacts_dir: os.PathLike | str | None = None):
    path = Path(artifacts_dir or get_artifacts_dir()) / filename
    path.touch()
    return path


def get_dataset_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("dataset_ids.yaml", artifacts_dir)


def get_compile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("compile-jobs.yaml", artifacts_dir)


def get_profile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("profile-jobs.yaml", artifacts_dir)


def get_inference_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("inference-jobs.yaml", artifacts_dir)


def get_quantize_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("quantize-jobs.yaml", artifacts_dir)


def get_accuracy_file() -> Path:
    filepath = get_artifact_filepath("accuracy.csv")
    if filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            f.write("Model,Runtime,Torch Accuracy,Device Accuracy")
            for i in range(5):
                f.write(f",PSNR_{i}")
            f.write("\n")
    return filepath


def mock_tabulate_fn(df: pd.DataFrame, **kwargs):
    psnr_str = ""
    for i, (_, value) in enumerate(df.iterrows()):
        psnr_str += f",{value.psnr}"
        if i == 4:
            break
    return psnr_str, tabulate(df, **kwargs)


def get_and_sync_datasets_cache_dir(
    has_channel_transpose: bool, dataset_name: str, split_size: int
) -> Path:
    folder_name = "hub_datasets"
    if not has_channel_transpose:
        folder_name += "_nt"
    dir_path = get_artifacts_dir() / folder_name / dataset_name
    if dir_path.exists():
        return dir_path
    with qaihm_temp_dir() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_ids = load_yaml(get_dataset_ids_file())
        input_key, gt_key = get_val_dataset_id_keys(dataset_name, has_channel_transpose)
        with open(tmp_path / f"split_size_{split_size}.txt", "w") as f:
            f.write(dataset_ids[input_key] + " " + dataset_ids[gt_key])

        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        hub.get_dataset(dataset_ids[input_key]).download(
            str(get_dataset_path(cache_path, dataset_ids[input_key]))
        )
        hub.get_dataset(dataset_ids[gt_key]).download(
            str(get_dataset_path(cache_path, dataset_ids[gt_key]))
        )
        with open(cache_path / "current_split_size.txt", "w") as f:
            f.write(str(split_size) + "\n")

        dir_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_path, dir_path)
    return dir_path


def append_line_to_file(path: Path, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def mock_get_calibration_data(
    input_spec: InputSpec, dataset_name: str, num_samples: int
) -> hub.Dataset:
    cache_key = dataset_name + "_train"
    dataset_ids_file = get_dataset_ids_file()
    dataset_ids = load_yaml(dataset_ids_file)
    if dataset_ids and cache_key in dataset_ids:
        return hub.get_dataset(dataset_ids[cache_key])
    dataset = get_calibration_data(input_spec, dataset_name, num_samples)
    hub_dataset = hub.upload_dataset(dataset)
    append_line_to_file(dataset_ids_file, f"{cache_key}: {hub_dataset.dataset_id}")
    return hub_dataset


class EveryNSampler(Sampler):
    """Samples every N samples deterministically from a torch dataset."""

    def __init__(self, n: int, num_samples: int):
        self.n = n
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(0, self.num_samples * self.n, self.n))

    def __len__(self):
        return self.num_samples


def get_val_dataset_id_keys(
    dataset_name: str, apply_channel_transpose: bool
) -> tuple[str, str]:
    base_name = f"{dataset_name}_val_"
    if not apply_channel_transpose:
        base_name += "nt_"  # no transpose
    return base_name + "input", base_name + "gt"


def get_torch_val_dataloader(dataset_name: str, num_samples: int) -> DataLoader:
    """
    Creates a torch dataloader with a chunk of validation data from the given dataset.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters:
        dataset_name: Name of the dataset. Dataset must be registered in
            qai_hub_models.datasets.__init__.py
        num_samples: Number of samples to sample from the full dataset.
    """
    torch_val_dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)
    sampler = EveryNSampler(
        n=len(torch_val_dataset) // num_samples, num_samples=num_samples
    )
    return DataLoader(torch_val_dataset, batch_size=num_samples, sampler=sampler)


def get_hub_val_dataset(
    dataset_name: str,
    ids_file: Path,
    model_cls: type[BaseModel],
    apply_channel_transpose: bool,
    num_samples: int,
) -> hub.Dataset:
    """
    Creates a hub dataset with a chunk of validation data from the given dataset.

    If the dataset ids are already present in the file, the function
    just loads and returns it.

    Otherwise, it creates the dataset and writes the ids to the file.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters:
        dataset_name: Name of the dataset. Dataset must be registered in
            qai_hub_models.datasets.__init__.py
        ids_file: Path to file where dataset ids are stored.
        model_cls: The model class using this data. Used to determine input spec
            and channel last inputs.
        apply_channel_transpose: If False, returns all inputs in channel first format.
            If True, applies channel last transpose for the inputs specified by the model.
        num_samples: Number of samples to sample from the full dataset.
    """
    dataset_ids = load_yaml(ids_file)
    input_key, gt_key = get_val_dataset_id_keys(dataset_name, apply_channel_transpose)
    if dataset_ids and input_key in dataset_ids:
        assert gt_key in dataset_ids
        return hub.get_dataset(dataset_ids[input_key])
    dataloader = get_torch_val_dataloader(dataset_name, num_samples)
    batch = next(iter(dataloader))
    input_entries, gt_entries = dataset_entries_from_batch(
        batch,
        list(model_cls.get_input_spec().keys()),
        model_cls.get_channel_last_inputs() if apply_channel_transpose else [],
    )

    input_dataset = hub.upload_dataset(input_entries)
    gt_dataset = hub.upload_dataset(gt_entries)
    append_line_to_file(ids_file, f"{input_key}: {input_dataset.dataset_id}")
    append_line_to_file(ids_file, f"{gt_key}: {gt_dataset.dataset_id}")
    return input_dataset
