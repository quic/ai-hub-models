# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Utilities for evaluating model accuracy on device.
"""

import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Sized, Tuple, Union

import h5py
import numpy as np
import qai_hub as hub
import torch
from qai_hub.public_rest_api import DatasetEntries
from qai_hub.util.dataset_entries_converters import dataset_entries_to_h5
from torch.utils.data import DataLoader, Dataset, random_split

from qai_hub_models.datasets import BaseDataset, get_dataset_from_name
from qai_hub_models.models.protocols import EvalModelProtocol
from qai_hub_models.utils.asset_loaders import (
    get_hub_datasets_path,
    load_h5,
    load_raw_file,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.inference import AsyncOnDeviceModel, make_hub_dataset_entries
from qai_hub_models.utils.qai_hub_helpers import transpose_channel_last_to_first

CACHE_SPLIT_SIZE_FILE = "current_split_size.txt"


def get_dataset_path(cache_path: Path, dataset_id: str) -> Path:
    return cache_path / f"dataset-{dataset_id}.h5"


def get_dataset_ids_filepath(dataset_name: str, split_size: int) -> Path:
    return get_hub_datasets_path() / dataset_name / f"split_size_{split_size}.txt"


def get_dataset_cache_filepath(dataset_name: str) -> Path:
    return get_hub_datasets_path() / dataset_name / "cache"


def get_dataset_cache_split_size(dataset_name: str) -> Optional[int]:
    """
    Get the split_size used for the current dataset cache.
    """
    path = get_dataset_cache_filepath(dataset_name) / CACHE_SPLIT_SIZE_FILE
    if not path.exists():
        return None
    curr_split_str = load_raw_file(path)
    return int(curr_split_str.strip())


def read_dataset_ids(
    dataset_ids_filepath: Union[str, Path]
) -> Tuple[List[str], List[str]]:
    input_ids = []
    gt_ids = []
    with open(dataset_ids_filepath, "r") as f:
        for line in f.readlines():
            input_id, gt_id = line.strip().split(" ")
            input_ids.append(input_id)
            gt_ids.append(gt_id)
    return input_ids, gt_ids


def write_entries_to_file(
    filename: Union[str, Path], dataset_entries: DatasetEntries
) -> None:
    """
    Write dataset entries to a .h5 file.
    """
    with h5py.File(filename, "w") as h5f:
        dataset_entries_to_h5(dataset_entries, h5f)


def _validate_dataset_ids_path(path: Path) -> bool:
    if not path.exists():
        return False
    input_ids, gt_ids = read_dataset_ids(path)

    # If any dataset is expired, evict the cache
    for dataset_id in input_ids + gt_ids:
        dataset = hub.get_dataset(dataset_id)
        if dataset.get_expiration_status() == "Expired":
            os.remove(path)
            return False
    return True


def _validate_inputs(num_batches: int) -> None:
    if num_batches < 1 and num_batches != -1:
        raise ValueError("num_batches must be positive or -1.")


def _populate_data_cache_impl(
    dataset: BaseDataset,
    split_size: int,
    seed: int,
    input_names: List[str],
    channel_last_input: Optional[str],
    cache_path: Path,
    dataset_ids_filepath: Path,
) -> None:
    """
    Does the heavy lifting of uploading data to hub and also saving it locally.
    """
    torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=split_size, shuffle=True)
    for sample in dataloader:
        model_inputs, ground_truth_values, *_ = sample
        if isinstance(ground_truth_values, tuple):
            output_names = [f"output_{i}" for i in range(len(ground_truth_values))]
        else:
            output_names = ["output_0"]
        input_entries = make_hub_dataset_entries(
            input_names,
            channel_last_input,
            TargetRuntime.TFLITE,
            model_inputs.split(1, dim=0),
        )
        gt_entries = make_hub_dataset_entries(
            output_names, None, TargetRuntime.TFLITE, ground_truth_values
        )
        # print(input_entries)
        input_dataset = hub.upload_dataset(input_entries)
        gt_dataset = hub.upload_dataset(gt_entries)
        input_data_path = get_dataset_path(cache_path, input_dataset.dataset_id)
        gt_data_path = get_dataset_path(cache_path, gt_dataset.dataset_id)
        write_entries_to_file(input_data_path, input_entries)
        write_entries_to_file(gt_data_path, gt_entries)
        with open(dataset_ids_filepath, "a") as f:
            f.write(f"{input_dataset.dataset_id} {gt_dataset.dataset_id}\n")


def _populate_data_cache(
    dataset: BaseDataset,
    split_size: int,
    seed: int,
    input_names: List[str],
    channel_last_input: Optional[str],
) -> None:
    """
    Creates hub datasets out of the input dataset and stores the same data locally.

    Divides the input dataset into subsets of size `split_size` and creates a separate
    hub dataset for each subset.

    Creates a local file `split_size_{split_size}.txt` that stores the current timestamp
    and all the dataset ids for both the inputs and labels. Datasets expire after 30
    days, so this file will need to be refreshed with a new dataset then.

    The data will also be stored locally in a "cache" folder. If this function is called
    with a new split size, the existing local data is deleted and replaced with the
    datasets made with the new split size.

    If called again with an old split size that still has its
    `split_size_{split_size}.txt` present, the data from those ids will
    be downloaded to the local cache instead of creating new hub datasets.

    Parameters:
        dataset: The torch dataset with the samples to cache.
        split_size: The maximum size of each hub.Dataset.
        seed: The random seed used to create the splits.
        input_names: The input names of the model.
        channel_last_input:
            Comma separated list of input names to have channel transposed.
    """
    dataset_name = dataset.__class__.dataset_name()
    os.makedirs(get_hub_datasets_path() / dataset_name, exist_ok=True)
    dataset_ids_path = get_dataset_ids_filepath(dataset_name, split_size)
    dataset_ids_valid = _validate_dataset_ids_path(dataset_ids_path)
    cache_path = get_dataset_cache_filepath(dataset_name)
    if dataset_ids_valid and get_dataset_cache_split_size(dataset_name) == split_size:
        print("Cached data already present.")
        return

    if cache_path.exists():
        shutil.rmtree(cache_path)

    with qaihm_temp_dir() as tmp_dir:
        tmp_cache_path = Path(tmp_dir) / "cache"
        os.makedirs(tmp_cache_path)
        if not dataset_ids_valid:
            tmp_dataset_ids_path = Path(tmp_dir) / "dataset_ids.txt"
            _populate_data_cache_impl(
                dataset,
                split_size,
                seed,
                input_names,
                channel_last_input,
                tmp_cache_path,
                tmp_dataset_ids_path,
            )
            shutil.move(str(tmp_dataset_ids_path), str(dataset_ids_path))
        else:
            for input_id, gt_id in zip(*read_dataset_ids(dataset_ids_path)):
                hub.get_dataset(input_id).download(
                    str(get_dataset_path(tmp_cache_path, input_id))
                )
                hub.get_dataset(gt_id).download(
                    str(get_dataset_path(tmp_cache_path, gt_id))
                )
        with open(tmp_cache_path / CACHE_SPLIT_SIZE_FILE, "w") as f:
            f.write(str(split_size) + "\n")
        shutil.move(str(tmp_cache_path), str(cache_path))


def sample_dataset(dataset: Dataset, num_samples: int, seed: int) -> Dataset:
    """
    Create a dataset that is a subsample of `dataset` with `num_samples`.

    Parameters:
        dataset: Original dataset with all data.
        num_samples: Number of samples in dataset subset.
        seed: Random seed to use when choosing the subsample.

    Returns:
        Sampled dataset.
    """
    assert isinstance(dataset, Sized), "Dataset must implement __len__."
    n = len(dataset)
    if num_samples == -1 or num_samples >= n:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    sampled_dataset, _ = random_split(
        dataset, [num_samples, n - num_samples], generator
    )
    return sampled_dataset


class HubDataset(Dataset):
    """
    Class the behaves like a PyTorch dataset except it is populated with hub datasets.

    Each returned batch corresponds to the data in a hub dataset
        that has been downloaded locally.
    """

    def __init__(
        self, dataset_name: str, num_samples: int, channel_last_input: Optional[str]
    ):
        self.cache_path = get_dataset_cache_filepath(dataset_name)
        self.split_size = get_dataset_cache_split_size(dataset_name)
        assert self.split_size is not None, "Dataset cache must be pre-populated"
        dataset_ids_filepath = get_dataset_ids_filepath(dataset_name, self.split_size)
        self.input_ids, self.gt_ids = read_dataset_ids(dataset_ids_filepath)

        max_splits = len(self.input_ids)
        if num_samples == -1:
            self.num_splits = max_splits
        else:
            self.num_splits = int(np.ceil(num_samples / self.split_size))
            self.num_splits = min(self.num_splits, max_splits)

        # Only print the warning when doing a partial dataset
        if self.num_splits < max_splits:
            if num_samples != self.num_splits * self.split_size:
                print(
                    "Rounding up number of samples to the nearest multiple of "
                    f" {self.split_size}: {num_samples} -> "
                    f"{self.num_splits * self.split_size}."
                )
        self.channel_last_input = channel_last_input

    def __len__(self) -> int:
        return self.num_splits

    def __getitem__(self, index) -> Any:
        input_data = load_h5(get_dataset_path(self.cache_path, self.input_ids[index]))
        if self.channel_last_input:
            input_data = transpose_channel_last_to_first(
                self.channel_last_input, input_data, TargetRuntime.TFLITE
            )
        input_np_data = np.concatenate(list(input_data.values())[0], axis=0)
        gt_data = load_h5(get_dataset_path(self.cache_path, self.gt_ids[index]))
        gt_torch_data = []
        for gt_np_tensor in gt_data.values():
            gt_torch_data.append(torch.from_numpy(np.concatenate(gt_np_tensor, axis=0)))

        gt_ret = gt_torch_data[0] if len(gt_torch_data) == 1 else tuple(gt_torch_data)
        return torch.from_numpy(input_np_data), gt_ret


def _make_dataloader(dataset: Dataset, seed: int, split_size: int) -> DataLoader:
    if isinstance(dataset, HubDataset):
        # Each batch should be direct output of __getitem__ without additional batch dim
        def _collate_fn(x):
            return x[0]

        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    torch.manual_seed(seed)
    return DataLoader(dataset, batch_size=split_size, shuffle=True)


def evaluate_on_dataset(
    compiled_model: hub.Model,
    torch_model: BaseModel,
    hub_device: hub.Device,
    dataset_name: str,
    split_size: int,
    num_samples: int,
    seed: int = 42,
    profile_options: str = "",
    use_cache: bool = False,
) -> Tuple[str, str]:
    """
    Evaluate model accuracy on a dataset both on device and with PyTorch.

    Parameters:
        compiled_model: A hub.Model object pointing to compiled model on hub.
            This is what will be used to compute accuracy on device.
        torch_model: The torch model to evaluate locally to compare accuracy.
        hub_device: Which device to use for on device measurement.
        dataset_name: The name of the dataset to use for evaluation.
        split_size: Limit on the number of samples to submit in a single inference job.
        num_samples: The number of samples to use for evaluation.
        seed: The random seed to use when subsampling the dataset.
        profile_options: Options to set when running inference on device.
            For example, which compute unit to use.
        use_cache: If set, will upload the full dataset to hub and store a local copy.
            This prevents re-uploading data to hub for each evaluation, with the
            tradeoff of increased initial overhead.

    Returns:
        Tuple of (torch accuracy, on device accuracy) both as formatted strings.
    """
    assert isinstance(torch_model, EvalModelProtocol), "Model must have an evaluator."
    _validate_inputs(num_samples)

    source_torch_dataset = get_dataset_from_name(dataset_name)
    input_names = list(torch_model.get_input_spec().keys())
    on_device_model = AsyncOnDeviceModel(
        compiled_model, input_names, hub_device, profile_options
    )

    torch_dataset: Dataset
    if use_cache:
        _populate_data_cache(
            source_torch_dataset,
            split_size,
            seed,
            on_device_model.input_names,
            on_device_model.channel_last_input,
        )
        torch_dataset = HubDataset(
            dataset_name, num_samples, on_device_model.channel_last_input
        )
    else:
        torch_dataset = sample_dataset(source_torch_dataset, num_samples, seed)
    dataloader = _make_dataloader(torch_dataset, seed, split_size)

    torch_evaluator = torch_model.get_evaluator()
    on_device_evaluator = torch_model.get_evaluator()

    on_device_results = []
    num_batches = len(dataloader)
    for i, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample

        if isinstance(torch_dataset, HubDataset):
            hub_dataset = hub.get_dataset(torch_dataset.input_ids[i])
            on_device_results.append(on_device_model(hub_dataset))
        else:
            on_device_results.append(on_device_model(model_inputs.split(1, dim=0)))

        for model_input, ground_truth in zip(model_inputs, ground_truth_values):
            torch_output = torch_model(model_input.unsqueeze(0))
            torch_evaluator.add_batch(torch_output, ground_truth.unsqueeze(0))
        print(
            f"Cumulative torch accuracy on batch {i + 1}/{num_batches}: "
            f"{torch_evaluator.formatted_accuracy()}"
        )

    dataloader = _make_dataloader(torch_dataset, seed, split_size)
    for i, sample in enumerate(dataloader):
        on_device_values = on_device_results[i].wait()
        _, ground_truth_values, *_ = sample
        on_device_evaluator.add_batch(on_device_values, ground_truth_values)
        print(
            f"Cumulative on device accuracy on batch {i + 1}/{num_batches}: "
            f"{on_device_evaluator.formatted_accuracy()}"
        )
    torch_accuracy = torch_evaluator.formatted_accuracy()
    on_device_accuracy = on_device_evaluator.formatted_accuracy()

    print("\nFinal accuracy:")
    print(f"torch: {torch_accuracy}")
    print(f"on-device: {on_device_accuracy}")
    return (torch_accuracy, on_device_accuracy)
