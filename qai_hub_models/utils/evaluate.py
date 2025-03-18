# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Utilities for evaluating model accuracy on device.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
import onnxruntime
import qai_hub as hub
import torch
from qai_hub.public_rest_api import DatasetEntries
from qai_hub.util.dataset_entries_converters import dataset_entries_to_h5
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm

from qai_hub_models.datasets import BaseDataset, DatasetSplit, get_dataset_from_name
from qai_hub_models.models.protocols import EvalModelProtocol
from qai_hub_models.utils.asset_loaders import (
    get_hub_datasets_path,
    load_h5,
    load_raw_file,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.onnx_helpers import mock_torch_onnx_inference
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first

CACHE_SPLIT_SIZE_FILE = "current_split_size.txt"
DEFAULT_NUM_EVAL_SAMPLES = 1000


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
) -> tuple[list[str], list[str]]:
    input_ids = []
    gt_ids = []
    with open(dataset_ids_filepath) as f:
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


class EveryNSampler(Sampler):
    """Samples every N samples deterministically from a torch dataset."""

    def __init__(self, n: int, num_samples: int):
        self.n = n
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        return iter(range(0, self.num_samples * self.n, self.n))

    def __len__(self) -> int:
        return self.num_samples


def get_deterministic_sample(
    dataset: BaseDataset, num_samples: int, split_size: int | None = None
) -> DataLoader:
    """
    Creates a torch dataloader with a subset validation data from the given dataset.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters:
        dataset_name: Name of the dataset. Dataset must be registered in
            qai_hub_models.datasets.__init__.py
        num_samples: Number of samples to sample from the full dataset.
        split_size: The batch size for the dataloader. If not set, all samples
            will be in the first batch.
    """
    split_size = split_size or num_samples
    sampler = EveryNSampler(n=len(dataset) // num_samples, num_samples=num_samples)
    return DataLoader(dataset, batch_size=split_size, sampler=sampler)


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
    return get_deterministic_sample(torch_val_dataset, num_samples)


def get_qdq_onnx(model: hub.Model) -> hub.Model | None:
    """
    Extracts the qdq model from the source quantize job.

    If the model was not ultimately from a quantize job, returns None.
    """
    if not isinstance(model.producer, hub.CompileJob):
        return None
    if not isinstance(model.producer.model, hub.Model):
        return None
    if not isinstance(model.producer.model.producer, hub.QuantizeJob):
        return None
    return model.producer.model


def _make_quant_cpu_session(model: hub.Model) -> onnxruntime.InferenceSession:
    """
    Creates an onnx runtime session with the qdq onnx model that was used to produce
    this hub.Model. Assumes the model was produced by a compile job, and the source
    model for the compile job was from a quantize job.
    """
    qdq_model = get_qdq_onnx(model)
    assert qdq_model is not None, "Model must be from a quantize job."
    local_dir = Path("build/qdq_cache_dir")
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / f"{qdq_model.model_id}.onnx"
    if not local_path.exists():
        qdq_model.download(str(local_path))
    return onnxruntime.InferenceSession(local_path)


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


def _validate_inputs(num_samples: int, dataset: BaseDataset) -> None:
    if num_samples < 1 and num_samples != -1:
        raise ValueError("num_samples must be positive or -1.")

    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} samples when dataset only has {len(dataset)}."
        )


def _populate_data_cache_impl(
    dataset: BaseDataset,
    split_size: int,
    seed: Optional[int],
    input_names: list[str],
    channel_last_input: Optional[list[str]],
    cache_path: Path,
    dataset_ids_filepath: Path,
) -> None:
    """
    Does the heavy lifting of uploading data to hub and also saving it locally.
    """
    if seed is not None:
        torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=split_size, shuffle=True)
    for batch in dataloader:
        input_entries, gt_entries = dataset_entries_from_batch(
            batch, input_names, channel_last_input
        )
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
    seed: Optional[int],
    input_names: list[str],
    channel_last_input: Optional[list[str]],
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


def sample_dataset(dataset: Dataset, num_samples: int, seed: int = 42) -> Dataset:
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
        self,
        dataset_name: str,
        num_samples: int,
        channel_last_input: Optional[list[str]],
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
                self.channel_last_input, input_data
            )
        input_np_data = np.concatenate(list(input_data.values())[0], axis=0)
        gt_data = load_h5(get_dataset_path(self.cache_path, self.gt_ids[index]))
        gt_torch_data = []
        for gt_np_tensor in gt_data.values():
            gt_torch_data.append(torch.from_numpy(np.concatenate(gt_np_tensor, axis=0)))

        gt_ret = gt_torch_data[0] if len(gt_torch_data) == 1 else tuple(gt_torch_data)
        return torch.from_numpy(input_np_data), gt_ret


def _make_dataloader(dataset: Dataset, split_size: int) -> DataLoader:
    if isinstance(dataset, HubDataset):
        # Each batch should be direct output of __getitem__ without additional batch dim
        def _collate_fn(x):
            return x[0]

        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    return DataLoader(dataset, batch_size=split_size, shuffle=False)


def evaluate_on_dataset(
    compiled_model: hub.Model,
    torch_model: BaseModel,
    hub_device: hub.Device,
    dataset_name: str,
    split_size: int,
    num_samples: int,
    seed: Optional[int] = None,
    profile_options: str = "",
    use_cache: bool = False,
    compute_quant_cpu_accuracy: bool = False,
    skip_device_accuracy: bool = False,
) -> tuple[float, Optional[float], Optional[float]]:
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
        seed: The random seed to use when subsampling the dataset. If not set, creates
            a deterministic subset.
        profile_options: Options to set when running inference on device.
            For example, which compute unit to use.
        use_cache: If set, will upload the full dataset to hub and store a local copy.
            This prevents re-uploading data to hub for each evaluation, with the
            tradeoff of increased initial overhead.
        compute_quant_cpu_accuracy: If the compiled model came from a quantize job,
            and this option is set, computes the quantized ONNX accuracy on the CPU.
        skip_device_accuracy: If set, skips on-device accuracy checks.

    Returns:
        Tuple of (torch accuracy, quant cpu accuracy, on device accuracy) all as float.
        quant cpu accuracy is the accuracy from running the quantized ONNX on the CPU.
        If quant cpu accuracy was not computed, its value in the tuple will be None.
    """
    assert isinstance(torch_model, EvalModelProtocol), "Model must have an evaluator."
    source_torch_dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)

    _validate_inputs(num_samples, source_torch_dataset)
    input_names = list(torch_model.get_input_spec().keys())
    output_names = torch_model.get_output_names()
    on_device_model = AsyncOnDeviceModel(
        compiled_model, input_names, hub_device, profile_options, output_names
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
        dataloader = _make_dataloader(torch_dataset, split_size)
    else:
        if seed is not None:
            torch_dataset = sample_dataset(source_torch_dataset, num_samples, seed)
            dataloader = _make_dataloader(torch_dataset, split_size)
        else:
            dataloader = get_deterministic_sample(
                source_torch_dataset, num_samples, split_size
            )

    torch_evaluator = torch_model.get_evaluator()
    on_device_evaluator = torch_model.get_evaluator()
    quant_cpu_evaluator = (
        torch_model.get_evaluator() if compute_quant_cpu_accuracy else None
    )
    quant_cpu_session = (
        _make_quant_cpu_session(compiled_model) if compute_quant_cpu_accuracy else None
    )

    on_device_results = []
    num_batches = len(dataloader)
    for i, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample

        if use_cache:
            assert isinstance(dataloader.dataset, HubDataset)
            hub_dataset = hub.get_dataset(dataloader.dataset.input_ids[i])
            if not skip_device_accuracy:
                on_device_results.append(on_device_model(hub_dataset))
        elif not skip_device_accuracy:
            on_device_results.append(on_device_model(model_inputs.split(1, dim=0)))

        for j, model_input in tqdm(enumerate(model_inputs), total=len(model_inputs)):
            if isinstance(ground_truth_values, torch.Tensor):
                ground_truth = ground_truth_values[j : j + 1]
            else:
                ground_truth = tuple(val[j : j + 1] for val in ground_truth_values)
            torch_input = model_input.unsqueeze(0)
            torch_output = torch_model(torch_input)
            torch_evaluator.add_batch(torch_output, ground_truth)
            if quant_cpu_evaluator is not None and quant_cpu_session is not None:
                quant_cpu_out = mock_torch_onnx_inference(
                    quant_cpu_session, torch_input
                )
                quant_cpu_evaluator.add_batch(quant_cpu_out, ground_truth)

        if quant_cpu_evaluator is not None:
            print(
                f"Cumulative quant cpu accuracy on batch {i + 1}/{num_batches}: "
                f"{quant_cpu_evaluator.formatted_accuracy()}"
            )
        print(
            f"Cumulative torch accuracy on batch {i + 1}/{num_batches}: "
            f"{torch_evaluator.formatted_accuracy()}"
        )

    if not skip_device_accuracy:
        for i, sample in enumerate(dataloader):
            on_device_values = on_device_results[i].wait()
            _, ground_truth_values, *_ = sample
            on_device_evaluator.add_batch(on_device_values, ground_truth_values)
            print(
                f"Cumulative on device accuracy on batch {i + 1}/{num_batches}: "
                f"{on_device_evaluator.formatted_accuracy()}"
            )
    torch_accuracy = torch_evaluator.formatted_accuracy()
    on_device_accuracy = None
    quant_cpu_acc = None
    print("\nFinal accuracy:")
    print(f"torch: {torch_accuracy}")
    if quant_cpu_evaluator is not None:
        quant_cpu_acc = quant_cpu_evaluator.get_accuracy_score()
        print(f"quant cpu: {quant_cpu_evaluator.formatted_accuracy()}")
    if not skip_device_accuracy:
        on_device_accuracy = on_device_evaluator.get_accuracy_score()
        print(f"on-device: {on_device_evaluator.formatted_accuracy()}")
    return (
        torch_evaluator.get_accuracy_score(),
        quant_cpu_acc,
        on_device_accuracy,
    )
