# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Utilities for evaluating model accuracy on device.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Callable, Optional, Union, cast

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
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.protocols import EvalModelProtocol, ExecutableModelProtocol
from qai_hub_models.utils.asset_loaders import (
    get_hub_datasets_path,
    load_h5,
    load_raw_file,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.onnx_helpers import extract_io_types_from_onnx_model
from qai_hub_models.utils.onnx_torch_wrapper import (
    OnnxModelTorchWrapper,
    OnnxSessionTorchWrapper,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first

CACHE_SAMPLES_PER_JOB_FILE = "current_samples_per_job.txt"
DEFAULT_NUM_EVAL_SAMPLES = 1000


@dataclass
class EvaluateResult:
    torch_accuracy: float | None = None
    sim_accuracy: float | None = None
    device_accuracy: float | None = None


def get_dataset_path(cache_path: Path, dataset_id: str) -> Path:
    return cache_path / f"dataset-{dataset_id}.h5"


def get_dataset_ids_filepath(dataset_name: str, samples_per_job: int) -> Path:
    return (
        get_hub_datasets_path()
        / dataset_name
        / f"samples_per_job_{samples_per_job}.txt"
    )


def get_dataset_cache_filepath(dataset_name: str) -> Path:
    return get_hub_datasets_path() / dataset_name / "cache"


def get_dataset_cache_samples_per_job(dataset_name: str) -> Optional[int]:
    """
    Get the samples_per_job used for the current dataset cache.
    """
    path = get_dataset_cache_filepath(dataset_name) / CACHE_SAMPLES_PER_JOB_FILE
    if not path.exists():
        return None
    curr_samples_per_job_str = load_raw_file(path)
    return int(curr_samples_per_job_str.strip())


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
    dataset: BaseDataset, num_samples: int, samples_per_job: int | None = None
) -> DataLoader:
    """
    Creates a torch dataloader with a subset validation data from the given dataset.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters:
        dataset_name: Name of the dataset. Dataset must be registered in
            qai_hub_models.datasets.__init__.py
        num_samples: Number of samples to sample from the full dataset.
        samples_per_job: The batch size for the dataloader. If not set, all samples
            will be in the first batch.
    """
    samples_per_job = samples_per_job or num_samples
    if num_samples < len(dataset) and num_samples != -1:
        sampler = EveryNSampler(n=len(dataset) // num_samples, num_samples=num_samples)
    else:
        sampler = None
    return DataLoader(dataset, batch_size=samples_per_job, sampler=sampler)


def get_torch_val_dataloader(
    dataset_name: str, num_samples: int | None = None
) -> DataLoader:
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
    num_samples = num_samples or torch_val_dataset.default_samples_per_job()
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


def _load_quant_cpu_onnx(model: hub.Model) -> OnnxModelTorchWrapper:
    """
    Creates an onnx runtime session with the qdq onnx model that was used to produce
    this hub.Model. Assumes the model was produced by a compile job, and the source
    model for the compile job was from a quantize job. Throws an exception otherwise.
    """
    qdq_model = get_qdq_onnx(model)
    assert qdq_model is not None, "Model must be from a quantize job."
    local_dir = Path("build/qdq_cache_dir")
    local_dir.mkdir(exist_ok=True)
    local_path = local_dir / f"{qdq_model.model_id}.onnx"
    if not local_path.exists():
        qdq_model.download(str(local_path))
    return OnnxModelTorchWrapper.OnCPU(local_path)


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
    samples_per_job: int,
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
    dataloader = DataLoader(dataset, batch_size=samples_per_job, shuffle=True)
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
    samples_per_job: int,
    seed: Optional[int],
    input_names: list[str],
    channel_last_input: Optional[list[str]],
) -> None:
    """
    Creates hub datasets out of the input dataset and stores the same data locally.

    Divides the input dataset into subsets of size `samples_per_job` and creates a separate
    hub dataset for each subset.

    Creates a local file `samples_per_job_{samples_per_job}.txt` that stores the current timestamp
    and all the dataset ids for both the inputs and labels. Datasets expire after 30
    days, so this file will need to be refreshed with a new dataset then.

    The data will also be stored locally in a "cache" folder. If this function is called
    with a new samples_per_job, the existing local data is deleted and replaced with the
    datasets made with the new samples_per_job.

    If called again with an old samples_per_job that still has its
    `samples_per_job_{samples_per_job}.txt` present, the data from those ids will
    be downloaded to the local cache instead of creating new hub datasets.

    Parameters:
        dataset: The torch dataset with the samples to cache.
        samples_per_job: The maximum size of each hub.Dataset.
        seed: The random seed used to create the splits.
        input_names: The input names of the model.
        channel_last_input:
            Comma separated list of input names to have channel transposed.
    """
    dataset_name = dataset.__class__.dataset_name()
    os.makedirs(get_hub_datasets_path() / dataset_name, exist_ok=True)
    dataset_ids_path = get_dataset_ids_filepath(dataset_name, samples_per_job)
    dataset_ids_valid = _validate_dataset_ids_path(dataset_ids_path)
    cache_path = get_dataset_cache_filepath(dataset_name)
    if (
        dataset_ids_valid
        and get_dataset_cache_samples_per_job(dataset_name) == samples_per_job
    ):
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
                samples_per_job,
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
        with open(tmp_cache_path / CACHE_SAMPLES_PER_JOB_FILE, "w") as f:
            f.write(str(samples_per_job) + "\n")
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


class DatasetFromIOTuples(Dataset):
    """
    Dataset that takes batched inputs and gts as tuples:
        tuple(Input of shape [ B, *dims ], Second Input of shape [B, *dims ], ...)
        tuple(GT of shape [ B, *dims ], Second GT of shape [B, *dims ], ...)

    and converts each to regular tensors (if the tuple has 1 element) when getting a dataset item.
    """

    def __init__(self, inputs: tuple[torch.Tensor, ...], gt: tuple[torch.Tensor, ...]):
        self.curr_input = inputs
        self.curr_gt = gt
        super().__init__()

    def __len__(self):
        return self.curr_input[0].shape[0]

    def __getitem__(
        self, index
    ) -> tuple[
        tuple[torch.Tensor, ...] | torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor
    ]:
        inputs = tuple(x[index] for x in self.curr_input)
        gt = tuple(x[index] for x in self.curr_gt)
        return (
            inputs[0] if len(inputs) == 1 else inputs,
            gt[0] if len(gt) == 1 else gt,
        )


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
        input_names: list[str],
        channel_last_input: Optional[list[str]],
    ):
        self.input_names = input_names
        self.cache_path = get_dataset_cache_filepath(dataset_name)
        self.samples_per_hub_dataset: int = cast(
            int, get_dataset_cache_samples_per_job(dataset_name)
        )
        assert (
            self.samples_per_hub_dataset is not None
        ), "Dataset cache must be pre-populated"
        dataset_ids_filepath = get_dataset_ids_filepath(
            dataset_name, self.samples_per_hub_dataset
        )
        self.input_ids, self.gt_ids = read_dataset_ids(dataset_ids_filepath)

        max_splits = len(self.input_ids)
        if num_samples == -1:
            self.num_hub_datasets = max_splits
        else:
            self.num_hub_datasets = int(
                np.ceil(num_samples / self.samples_per_hub_dataset)
            )
            self.num_hub_datasets = min(self.num_hub_datasets, max_splits)

        # Only print the warning when doing a partial dataset
        if self.num_hub_datasets < max_splits:
            if num_samples != self.num_hub_datasets * self.samples_per_hub_dataset:
                print(
                    "Rounding up number of samples to the nearest multiple of "
                    f" {self.samples_per_hub_dataset}: {num_samples} -> "
                    f"{self.num_hub_datasets * self.samples_per_hub_dataset}."
                )
        self.channel_last_input = channel_last_input
        self.curr_dataset: DatasetFromIOTuples | None = None
        self.curr_h5_idx = -1

    def __len__(self) -> int:
        return self.num_hub_datasets * self.samples_per_hub_dataset

    def load_hub_dataset_for_sample(self, sample_index) -> DatasetFromIOTuples:
        """
        Fetches the AI Hub dataset that stores the sample at the given index,
        and stores it in memory.

        Returns a tuple of:
            input batch (torch.Tensor) containing this index
            ground truth batch (torch.Tensor) containing this index
            sample index within the returned batch
        """
        h5_idx = math.floor(sample_index / self.samples_per_hub_dataset)
        if self.curr_h5_idx != h5_idx or self.curr_dataset is None:
            self.curr_h5_idx = h5_idx

            # Convert DatasetEntries from dict{
            #   "Input0": batch_0, batch_1, ...]
            #   "Input1": batch_0, abtch_1, ...]
            # }
            #
            # to tuple(input_0_all_batches, input_1_all_batches, ...)
            input_dataset_entries = load_h5(
                get_dataset_path(self.cache_path, self.input_ids[sample_index])
            )
            if self.channel_last_input:
                input_dataset_entries = transpose_channel_last_to_first(
                    self.channel_last_input, input_dataset_entries
                )
            curr_input = tuple(
                torch.as_tensor(np.concatenate(input_list, axis=0))
                for input_list in input_dataset_entries.values()
            )

            # Convert DatasetEntries from dict{
            #   "Output0": batch_0, batch_1, ...]
            #   "Output1": batch_0, abtch_1, ...]
            # }
            #
            # to tuple(input_0_all_batches, input_1_all_batches, ...)
            gt_dataset_entries = load_h5(
                get_dataset_path(self.cache_path, self.gt_ids[sample_index])
            )
            curr_gt = tuple(
                torch.as_tensor(np.concatenate(gt_list, axis=0))
                for gt_list in gt_dataset_entries.values()
            )
            self.curr_dataset = DatasetFromIOTuples(curr_input, curr_gt)
        return self.curr_dataset

    def __getitem__(
        self, index
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        curr_dataset = self.load_hub_dataset_for_sample(index)
        index_in_hub_dataset = index - (self.curr_h5_idx * self.samples_per_hub_dataset)
        return curr_dataset[index_in_hub_dataset]


def evaluate(
    dataloader: DataLoader,
    evaluator_func: Callable[..., BaseEvaluator],
    models: dict[str, ExecutableModelProtocol | AsyncOnDeviceModel],
    model_batch_size: int | None = None,
    verbose: bool = False,
) -> dict[str, BaseEvaluator]:
    """
    Evaluate model accuracy on a dataset both on device and with PyTorch.

    Parameters:
        dataloader:
            batched data loader

        evaluator_func:
            Function that returns a new evaluator instance to use for eval.

        models:
            dict[model identifier, model class]
            Models to evaluate.

        model_batch_size:
            If set, models will always execute with this batch size.
            If None, all models run with the dataloader batch size.

            Generally this is set to 1 because most models compile to a batch size of 1.
            We typically use a batch size of 1 locally because:
                * It's not slower than multi-batch inference on CPU.
                * It's usually required to run compiled models with fixed input shapes.

            The dataloader may have a different batch size to accomodate
            caching of Hub Datasets.

        verbose:
            If true, prints evaluation scores.

    Returns:
        dict[model identifier, eval result]
    """
    ai_hub_inference_models = {
        n: m for n, m in models.items() if isinstance(m, AsyncOnDeviceModel)
    }
    local_inference_models = {
        n: m for n, m in models.items() if not isinstance(m, AsyncOnDeviceModel)
    }

    evaluators = {name: evaluator_func() for name in models}
    ai_hub_async_inference_outputs: dict[str, list[AsyncOnDeviceResult]] = {
        name: []
        for name, model in models.items()
        if isinstance(model, AsyncOnDeviceModel)
    }
    batch_size = dataloader.batch_size or 1
    model_batch_size = model_batch_size or batch_size

    if batch_size % model_batch_size != 0:
        raise ValueError(
            "The model batch size must evenly divide the DataLoader's batch size. It is otherwise impossible to evaluate the whole dataset on the same batch size."
        )

    # Get each sample from the dataloader. Each sample has batch size batch_size.
    for batch_idx, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample

        def _torch_io_to_tuple(
            val: list | tuple | torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            """
            Convert torch model I/O of any type to a tuple of inputs / outputs.
            """
            if isinstance(val, tuple):
                return val
            elif isinstance(val, list):
                return tuple(val)
            return tuple([val])

        model_inputs = _torch_io_to_tuple(model_inputs)
        ground_truth_values = _torch_io_to_tuple(ground_truth_values)

        # On device output is computed on the entire batch,
        # Run all on-device models first.
        for model_name, async_model in ai_hub_inference_models.items():
            if isinstance(dataloader.dataset, HubDataset):
                # If the dataloader is a cached dataset, we can just use the cached Hub dataset
                # instead of uploading the inputs to Hub again.
                assert async_model.input_names == dataloader.dataset.input_names
                assert (
                    async_model.channel_last_input
                    == dataloader.dataset.channel_last_input
                )
                hub_dataset = hub.get_dataset(dataloader.dataset.input_ids[batch_idx])
                async_output = async_model(hub_dataset)
            else:
                device_inputs = (
                    (x.split(model_batch_size, dim=0) for x in model_inputs)
                    if model_batch_size
                    else model_inputs
                )
                async_output = async_model(*device_inputs)

            # On device output is computed asynchronously on AI Hub, so save it for later.
            ai_hub_async_inference_outputs[model_name].append(async_output)

        # Run the remaining local models separately.
        if len(local_inference_models) == 0:
            continue

        # Run local inference on smaller batch size
        local_dataset = DatasetFromIOTuples(model_inputs, ground_truth_values)
        local_dataloader = DataLoader(local_dataset, model_batch_size)
        for sample in tqdm(local_dataloader):
            local_model_inputs, local_ground_truth_values, *_ = sample

            # Run inference on this smaller batch, add to evaluator.
            for model_name, local_model in local_inference_models.items():
                if type(local_model_inputs) in [list, tuple]:
                    batch_output = local_model(*local_model_inputs)
                else:
                    batch_output = local_model(local_model_inputs)
                evaluators[model_name].add_batch(
                    batch_output, local_ground_truth_values
                )

        if verbose:
            for model_name in local_inference_models:
                print(
                    f"Cumulative {model_name} accuracy on {(batch_idx + 1) * batch_size} samples: "
                    f"{evaluators[model_name].formatted_accuracy()}"
                )

    # Collect on device accuracy
    if len(ai_hub_inference_models) > 0:
        for batch_idx, sample in enumerate(dataloader):
            _, ground_truth_values, *_ = sample
            for (
                model_name,
                batched_async_model_outputs,
            ) in ai_hub_async_inference_outputs.items():
                model_output = batched_async_model_outputs[batch_idx].wait()
                evaluators[model_name].add_batch(model_output, ground_truth_values)

                if verbose:
                    print(
                        f"Cumulative {model_name} accuracy on {(batch_idx + 1) * batch_size} samples: "
                        f"{evaluators[model_name].formatted_accuracy()}"
                    )

    if verbose:
        print("\nFinal Accuracy")
        for model_name, evaluator in evaluators.items():
            print(f"{model_name}: {evaluator.formatted_accuracy()}")

    return evaluators


def evaluate_on_dataset(
    compiled_model: hub.Model,
    torch_model: BaseModel | CollectionModel,
    hub_device: hub.Device,
    dataset_name: str,
    samples_per_job: int | None = None,
    num_samples: int | None = None,
    seed: Optional[int] = None,
    profile_options: str = "",
    use_cache: bool = False,
    compute_quant_cpu_accuracy: bool = False,
    skip_device_accuracy: bool = False,
    skip_torch_accuracy: bool = False,
) -> EvaluateResult:
    f"""
    Evaluate model accuracy on a dataset both on device and with PyTorch.

    Parameters:
        compiled_model: A hub.Model object pointing to compiled model on hub.
            This is what will be used to compute accuracy on device.
        torch_model: The torch model to evaluate locally to compare accuracy.
        hub_device: Which device to use for on device measurement.
        dataset_name: The name of the dataset to use for evaluation.
        samples_per_job: Limit on the number of samples to submit in a single inference job.
            If not specified, uses the default value set on the dataset.
        num_samples: The number of samples to use for evaluation.
            If not set, uses the minimum of the samples_per_job and {DEFAULT_NUM_EVAL_SAMPLES}
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
        skip_device_accuracy: If set, skips torch cpu accuracy checks.

    Returns:
        Tuple of (torch accuracy, quant cpu accuracy, on device accuracy) all as float.
        quant cpu accuracy is the accuracy from running the quantized ONNX on the CPU.
        If any accuracy was not computed, its value in the tuple will be None.
    """
    assert isinstance(torch_model, EvalModelProtocol), "Model must have an evaluator."
    assert isinstance(
        torch_model, BaseModel
    ), "Evaluation is not yet supported for CollectionModels."

    input_names = list(torch_model.get_input_spec().keys())
    output_names = torch_model.get_output_names()
    on_device_model = AsyncOnDeviceModel(
        compiled_model, input_names, hub_device, profile_options, output_names
    )

    source_torch_dataset = get_dataset_from_name(
        dataset_name, DatasetSplit.VAL, input_spec=on_device_model.get_input_spec()
    )
    samples_per_job = samples_per_job or source_torch_dataset.default_samples_per_job()
    num_samples = num_samples or min(samples_per_job, DEFAULT_NUM_EVAL_SAMPLES)
    _validate_inputs(num_samples, source_torch_dataset)

    if use_cache:
        _populate_data_cache(
            source_torch_dataset,
            samples_per_job,
            seed,
            on_device_model.input_names,
            on_device_model.channel_last_input,
        )
        torch_dataset = HubDataset(
            dataset_name,
            num_samples,
            on_device_model.input_names,
            on_device_model.channel_last_input,
        )
        dataloader = DataLoader(
            torch_dataset, batch_size=samples_per_job, shuffle=False
        )
        num_samples = len(torch_dataset)
    else:
        if seed is not None:
            torch_dataset = sample_dataset(source_torch_dataset, num_samples, seed)
            dataloader = DataLoader(
                torch_dataset, batch_size=samples_per_job, shuffle=False
            )
        else:
            dataloader = get_deterministic_sample(
                source_torch_dataset, num_samples, samples_per_job
            )

    print(f"Evaluating on {num_samples} samples.")

    models: dict[str, ExecutableModelProtocol | AsyncOnDeviceModel] = {}
    if compute_quant_cpu_accuracy:
        models["quant cpu"] = _load_quant_cpu_onnx(compiled_model)
    if not skip_device_accuracy:
        models["on-device"] = on_device_model
    if not skip_torch_accuracy:
        models["torch"] = torch_model

    results = evaluate(
        dataloader,
        torch_model.get_evaluator,
        models,
        model_batch_size=1,
        verbose=True,
    )

    scores = {name: e.get_accuracy_score() for name, e in results.items()}
    return EvaluateResult(
        scores.get("torch"),
        scores.get("quant cpu"),
        scores.get("on-device"),
    )


@unique
class EvalMode(Enum):
    description: str

    FP = ("fp", "run floating point model")
    QUANTSIM = ("quantsim", "simulated quantization")
    ON_DEVICE = ("on-device", "physical device via AI Hub (slow)")
    LOCAL_DEVICE = ("local-device", "running on local device like X Elite")

    def __new__(cls, value: str, description: str):
        # object.__new__ so we bypass Enum.__init__ machinery
        obj = object.__new__(cls)
        obj._value_ = value  # this is the “real” .value
        obj.description = description  # store your help‐text
        return obj

    @staticmethod
    def from_string(string: str) -> EvalMode:
        key = string.replace("-", "_").upper()
        return EvalMode[key]

    def __str__(self):
        return self.value


def evaluate_session_on_dataset(
    session: onnxruntime.InferenceSession,
    torch_model: BaseModel | CollectionModel,
    dataset_name: str,
    num_samples: int | None = None,
) -> tuple[float, str]:
    f"""
    Evaluate model accuracy on a dataset using ONNX runtime.

    Parameters:
        session: ONNX session to evaluate.
        torch_model: The torch model to evaluate locally to compare accuracy.
        dataset_name: The name of the dataset to use for evaluation.
        num_samples: The number of samples to use for evaluation.
            If not set, uses the minimum of the samples_per_job and {DEFAULT_NUM_EVAL_SAMPLES}

    Returns:
        Tuple of accuracy(in float), formatted accuracy (as a string)
    """
    assert isinstance(torch_model, EvalModelProtocol), "Model must have an evaluator."
    assert isinstance(
        torch_model, BaseModel
    ), "Evaluation is not yet supported for CollectionModels."
    source_torch_dataset = get_dataset_from_name(dataset_name, DatasetSplit.VAL)
    num_samples = num_samples or DEFAULT_NUM_EVAL_SAMPLES

    _validate_inputs(num_samples, source_torch_dataset)

    dataloader = get_deterministic_sample(source_torch_dataset, num_samples, None)

    print(f"Evaluating on {num_samples} samples.")
    evaluator = torch_model.get_evaluator()
    inputs, outputs = extract_io_types_from_onnx_model(session)
    session_wrapper = OnnxSessionTorchWrapper(session, inputs, outputs)

    for i, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample

        for j, model_input in tqdm(enumerate(model_inputs), total=len(model_inputs)):
            if isinstance(ground_truth_values, torch.Tensor):
                ground_truth = ground_truth_values[j : j + 1]
            else:
                ground_truth = tuple(val[j : j + 1] for val in ground_truth_values)
            torch_input = model_input.unsqueeze(0)

            onnx_out = session_wrapper(torch_input)
            evaluator.add_batch(onnx_out, ground_truth)

    accuracy = evaluator.get_accuracy_score()
    formatted_accuracy = evaluator.formatted_accuracy()

    return accuracy, formatted_accuracy
