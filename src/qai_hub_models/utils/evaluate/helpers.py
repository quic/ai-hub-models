# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Utilities for evaluating model accuracy on device."""

from __future__ import annotations

import gc
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import onnxruntime
import qai_hub as hub
import torch
from qai_hub.public_rest_api import DatasetEntries
from qai_hub.util.dataset_entries_converters import dataset_entries_to_h5
from qai_hub_models_cli.errors import CLIUserError
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing_extensions import Self

from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.asset_loaders import (
    get_hub_datasets_path,
    load_h5,
    load_raw_file,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
    get_folder_name,
    instantiate_dataset,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.envvars import IsOnCIEnvvar
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import extract_io_types_from_onnx_model
from qai_hub_models.utils.onnx.torch_wrapper import (
    OnnxModelTorchWrapper,
    OnnxSessionOptions,
    OnnxSessionTorchWrapper,
)
from qai_hub_models.utils.private_asset_loaders import UnfetchableDatasetError
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first

CACHE_SAMPLES_PER_JOB_FILE = "current_samples_per_job.txt"
DEFAULT_NUM_EVAL_SAMPLES = 1000


def _count_num_samples(data: Sequence[torch.Tensor] | torch.Tensor) -> int:
    return len(data[0]) if isinstance(data, Sequence) else len(data)


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


def get_dataset_cache_filepath(folder_name: str) -> Path:
    return get_hub_datasets_path() / folder_name / "cache"


def get_dataset_cache_samples_per_job(folder_name: str) -> int | None:
    """Get the samples_per_job used for the current dataset cache."""
    path = get_dataset_cache_filepath(folder_name) / CACHE_SAMPLES_PER_JOB_FILE
    if not path.exists():
        return None
    curr_samples_per_job_str = load_raw_file(path)
    return int(curr_samples_per_job_str.strip())


def read_dataset_ids(
    dataset_ids_filepath: str | Path,
) -> tuple[list[str], list[str]]:
    input_ids = []
    gt_ids = []
    with open(dataset_ids_filepath) as f:
        for line in f:
            input_id, gt_id = line.strip().split(" ")
            input_ids.append(input_id)
            gt_ids.append(gt_id)
    return input_ids, gt_ids


def write_entries_to_file(
    filename: str | Path, dataset_entries: DatasetEntries
) -> None:
    """Write dataset entries to a .h5 file."""
    with h5py.File(filename, "w") as h5f:
        dataset_entries_to_h5(dataset_entries, h5f)


def get_deterministic_sample(
    dataset: BaseDataset, num_samples: int, samples_per_job: int | None = None
) -> DataLoader:
    """
    Creates a torch dataloader with a subset validation data from the given dataset.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters
    ----------
    dataset
        Dataset to sample from.
    num_samples
        Number of samples to sample from the full dataset.
    samples_per_job
        The batch size for the dataloader. If not set, all samples will be in the first batch.

    Returns
    -------
    dataloader : DataLoader
        Dataloader with sampled data.
    """
    return dataset.get_dataloader(num_samples, samples_per_job)


def get_torch_val_dataloader(
    dataset_cls: type[BaseDataset],
    num_samples: int | None = None,
    input_spec: InputSpec | None = None,
) -> DataLoader:
    """
    Creates a torch dataloader with a chunk of validation data from the given dataset.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    Parameters
    ----------
    dataset_cls
        Dataset class. Dataset must be registered in qai_hub_models.datasets.__init__.py
    num_samples
        Number of samples to sample from the full dataset.
    input_spec
        Input spec of the model requesting data.

    Returns
    -------
    dataloader : DataLoader
        Dataloader with validation data.
    """
    torch_val_dataset = instantiate_dataset(dataset_cls, DatasetSplit.VAL, input_spec)
    num_samples = num_samples or torch_val_dataset.default_samples_per_job()
    return get_deterministic_sample(torch_val_dataset, num_samples)


def get_qdq_onnx(model: hub.Model) -> hub.Model | None:
    """
    Extracts the qdq model from the source quantize job.

    If the model was not ultimately from a quantize job, returns None.

    Parameters
    ----------
    model
        Hub model to extract from.

    Returns
    -------
    qdq_model : hub.Model | None
        QDQ model, or None if not from a quantize job.
    """
    producer = model.get_producer()
    if isinstance(producer, hub.QuantizeJob):
        return model
    if not isinstance(producer, hub.CompileJob):
        return None
    if not isinstance(producer.model, hub.Model):
        return None
    if not isinstance(producer.model.get_producer(), hub.QuantizeJob):
        return None
    return producer.model


def _load_quant_cpu_onnx(model: hub.Model) -> OnnxModelTorchWrapper:
    """
    Creates an onnx runtime session with the qdq onnx model.

    Uses the qdq onnx model that was used to produce this hub.Model.
    Assumes the model was produced by a compile job, and the source
    model for the compile job was from a quantize job. Throws an exception otherwise.

    Parameters
    ----------
    model
        Hub model to load from.

    Returns
    -------
    wrapper : OnnxModelTorchWrapper
        ONNX model wrapper for CPU inference.
    """
    qdq_model = get_qdq_onnx(model)
    assert qdq_model is not None, "Model must be from a quantize job."
    local_dir = Path("build/qdq_cache_dir")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{qdq_model.model_id}"
    output_path = qdq_model.download(str(local_path))

    # Disable all graph optimizations to preserve the original quantized model structure as much as possible.
    # This is to workaround a bug in ONNXRT that causes an accuracy drop due to unsafe opt passes on quantized models
    options = OnnxSessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    return OnnxModelTorchWrapper.OnCPU(output_path, session_options=options)


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
        raise CLIUserError("num_samples must be positive or -1.")

    if num_samples > len(dataset):
        raise CLIUserError(
            f"Requested {num_samples} samples when dataset only has {len(dataset)}."
        )


def _populate_data_cache_impl(
    dataset: BaseDataset,
    samples_per_job: int,
    seed: int | None,
    input_names: list[str],
    channel_last_input: list[str] | None,
    cache_path: Path,
    dataset_ids_filepath: Path,
) -> None:
    """Does the heavy lifting of uploading data to hub and also saving it locally."""
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
    dataset_cls: type[BaseDataset],
    split: DatasetSplit,
    input_spec: InputSpec | None,
    samples_per_job: int,
    seed: int | None,
    input_names: list[str],
    channel_last_input: list[str] | None,
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

    Directory structure:

    hub_datasets{_nt}/
        {dataset_name}_{input_spec_hash}/
            samples_per_job_{num_samples}.txt  # Contains input and gt hub dataset ids
            cache/
                current_samples_per_job.txt  # Contains a single number of how many samples per job
                dataset-dxxxx.h5  # Raw data of what was uploaded to hub for input
                dataset-dyyyy.h5  # Raw data of what was uploaded to hub for gt

    Parameters
    ----------
    dataset_cls
        Dataset class to instantiate (if it is not cached).
    split
        Dataset split (train/val/test).
    input_spec
        Input spec for the model.
    samples_per_job
        The maximum size of each hub.Dataset.
    seed
        The random seed used to create the splits.
    input_names
        The input names of the model.
    channel_last_input
        Comma separated list of input names to have channel transposed.
    """
    dataset_name = dataset_cls.dataset_name()
    folder_name = get_folder_name(dataset_name, input_spec)
    os.makedirs(get_hub_datasets_path() / folder_name, exist_ok=True)
    dataset_ids_path = get_dataset_ids_filepath(folder_name, samples_per_job)
    dataset_ids_valid = _validate_dataset_ids_path(dataset_ids_path)
    cache_path = get_dataset_cache_filepath(folder_name)
    if (
        dataset_ids_valid
        and get_dataset_cache_samples_per_job(folder_name) == samples_per_job
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
            torch_dataset = instantiate_dataset(dataset_cls, split, input_spec)
            _populate_data_cache_impl(
                torch_dataset,
                samples_per_job,
                seed,
                input_names,
                channel_last_input,
                tmp_cache_path,
                tmp_dataset_ids_path,
            )
            shutil.move(str(tmp_dataset_ids_path), str(dataset_ids_path))
        else:
            for input_id, gt_id in zip(
                *read_dataset_ids(dataset_ids_path), strict=False
            ):
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

    Parameters
    ----------
    dataset
        Original dataset with all data.
    num_samples
        Number of samples in dataset subset.
    seed
        Random seed to use when choosing the subsample.

    Returns
    -------
    sampled_dataset : Dataset
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
    Dataset that takes batched inputs and gts as tuples.

    Takes tuples of the form:
        tuple(Input of shape [ B, *dims ], Second Input of shape [B, *dims ], ...)
        tuple(GT of shape [ B, *dims ], Second GT of shape [B, *dims ], ...)

    Converts each to regular tensors (if the tuple has 1 element) when getting a dataset item.

    Parameters
    ----------
    inputs
        Tuple of input tensors with batch dimension.
    gt
        Tuple of ground truth tensors with batch dimension.
    """

    def __init__(
        self, inputs: tuple[torch.Tensor, ...], gt: tuple[torch.Tensor, ...]
    ) -> None:
        self.curr_input = inputs
        self.curr_gt = gt
        super().__init__()

    def __len__(self) -> int:
        return self.curr_input[0].shape[0]

    def __getitem__(
        self, index: int
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
    Class that behaves like a PyTorch dataset except it is populated with hub datasets.

    Each returned batch corresponds to the data in a hub dataset
    that has been downloaded locally.

    Parameters
    ----------
    dataset_name
        Name of the dataset.
    num_samples
        Number of samples to include.
    input_names
        Names of the input tensors.
    channel_last_input
        List of input names that should be channel-last.
    input_spec
        Input specification for the model.
    """

    def __init__(
        self,
        dataset_name: str,
        num_samples: int,
        input_names: list[str],
        channel_last_input: list[str] | None,
        input_spec: InputSpec,
    ) -> None:
        self.input_names = input_names
        folder_name = get_folder_name(dataset_name, input_spec)
        self.cache_path = get_dataset_cache_filepath(folder_name)
        self.samples_per_hub_dataset: int = cast(
            int, get_dataset_cache_samples_per_job(folder_name)
        )
        assert self.samples_per_hub_dataset is not None, (
            "Dataset cache must be pre-populated"
        )
        dataset_ids_filepath = get_dataset_ids_filepath(
            folder_name, self.samples_per_hub_dataset
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
        if (
            self.num_hub_datasets < max_splits
            and num_samples != self.num_hub_datasets * self.samples_per_hub_dataset
        ):
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

    def load_hub_dataset_for_sample(self, sample_index: int) -> DatasetFromIOTuples:
        """
        Fetches the AI Hub Workbench dataset that stores the sample at the given index.

        Stores the dataset in memory.

        Parameters
        ----------
        sample_index
            Index of the sample to load.

        Returns
        -------
        dataset : DatasetFromIOTuples
            Dataset containing the input batch and ground truth batch for this index.
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
                get_dataset_path(self.cache_path, self.input_ids[self.curr_h5_idx])
            )
            if self.channel_last_input:
                input_dataset_entries = transpose_channel_last_to_first(
                    self.channel_last_input, input_dataset_entries
                )
            curr_input = tuple(
                torch.as_tensor(np.concatenate(input_list, axis=0))
                for input_list in input_dataset_entries.values()
            )
            input_dataset_entries = None  # type: ignore[assignment]
            gc.collect()  # delete h5 file data immediately to reduce memory pressure

            # Convert DatasetEntries from dict{
            #   "Output0": batch_0, batch_1, ...]
            #   "Output1": batch_0, abtch_1, ...]
            # }
            #
            # to tuple(input_0_all_batches, input_1_all_batches, ...)
            gt_dataset_entries = load_h5(
                get_dataset_path(self.cache_path, self.gt_ids[self.curr_h5_idx])
            )
            curr_gt = tuple(
                torch.as_tensor(np.concatenate(gt_list, axis=0))
                for gt_list in gt_dataset_entries.values()
            )
            gt_dataset_entries = None  # type: ignore[assignment]
            gc.collect()  # delete h5 file data immediately to reduce memory pressure

            self.curr_dataset = DatasetFromIOTuples(curr_input, curr_gt)

        return self.curr_dataset

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        curr_dataset = self.load_hub_dataset_for_sample(index)
        index_in_hub_dataset = index - (self.curr_h5_idx * self.samples_per_hub_dataset)
        return curr_dataset[index_in_hub_dataset]


def evaluate(
    dataloader: DataLoader,
    evaluator_func: Callable[..., BaseEvaluator],
    model_executors: Mapping[
        str, ExecutableModelProtocol | CollectionAppEvaluateProtocol
    ],
    model_batch_size: int | None = None,
    verbose: bool = False,
) -> dict[str, BaseEvaluator]:
    """
    Evaluate model accuracy on a dataset both on device and with PyTorch.

    Parameters
    ----------
    dataloader
        Batched data loader.
    evaluator_func
        Function that returns a new evaluator instance to use for eval.
    model_executors
        Models or apps to evaluate, keyed by model identifier.
    model_batch_size
        If set, models will always execute with this batch size.
        If None, all models run with the dataloader batch size.
        Generally this is set to 1 because most models compile to a batch size of 1.
        We typically use a batch size of 1 locally because:
        * It's not slower than multi-batch inference on CPU.
        * It's usually required to run compiled models with fixed input shapes.
        The dataloader may have a different batch size to accommodate caching of Hub Datasets.
    verbose
        If true, prints evaluation scores.

    Returns
    -------
    evaluators : dict[str, BaseEvaluator]
        Evaluator results, keyed by model identifier.
    """
    ai_hub_inference_models: dict[str, AsyncOnDeviceModel] = {}
    collection_inference_models: dict[str, CollectionAppEvaluateProtocol] = {}
    local_inference_models: dict[
        str, ExecutableModelProtocol | CollectionAppEvaluateProtocol
    ] = {}
    for n, m in model_executors.items():
        if isinstance(m, CollectionAppEvaluateProtocol) and m.uses_ondevice_model:
            collection_inference_models[n] = m
        elif isinstance(m, AsyncOnDeviceModel):
            ai_hub_inference_models[n] = m
        else:
            local_inference_models[n] = m

    def _get_evaluator(name: str, m: object) -> BaseEvaluator:
        if hasattr(m, "get_evaluator"):
            try:
                return m.get_evaluator()
            except NotImplementedError:
                pass
        return evaluator_func()

    evaluators = {name: _get_evaluator(name, m) for name, m in model_executors.items()}
    # A collection app whose last pipeline stage runs on CPU returns its output
    # directly instead of a pending job.
    async_outputs: dict[str, list[AsyncOnDeviceResult | tuple[torch.Tensor, ...]]] = {
        name: [] for name in {**ai_hub_inference_models, **collection_inference_models}
    }
    collection_model_generators: dict[str, list[CollectionModelEvalGenerator]] = (
        defaultdict(list)
    )
    batch_size = dataloader.batch_size or 1
    model_batch_size = model_batch_size or batch_size

    if batch_size % model_batch_size != 0:
        raise ValueError(
            "The model batch size must evenly divide the DataLoader's batch size. It is otherwise impossible to evaluate the whole dataset on the same batch size."
        )

    def _torch_io_to_tuple(
        val: list | tuple | torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Convert torch model I/O of any type to a tuple of inputs / outputs."""
        if isinstance(val, (list, tuple)):
            return tuple(val)
        return (val,)

    # Get each sample from the dataloader. Each sample has batch size batch_size.
    for batch_idx, sample in enumerate(dataloader):
        model_inputs, ground_truth_values, *_ = sample
        cumulative_samples = _count_num_samples(model_inputs) + batch_idx * batch_size

        model_inputs = _torch_io_to_tuple(model_inputs)
        ground_truth_values = _torch_io_to_tuple(ground_truth_values)

        # Run all on-device models first (output is computed asynchronously).
        for model_name, async_model_executor in ai_hub_inference_models.items():
            if isinstance(dataloader.dataset, HubDataset):
                # Use the cached Hub dataset instead of uploading inputs again.
                assert (
                    async_model_executor.input_names == dataloader.dataset.input_names
                )
                assert (
                    async_model_executor.channel_last_input
                    == dataloader.dataset.channel_last_input
                )
                hub_dataset = hub.get_dataset(dataloader.dataset.input_ids[batch_idx])
                async_output = async_model_executor(hub_dataset)
            else:
                device_inputs = (
                    (x.split(model_batch_size, dim=0) for x in model_inputs)
                    if model_batch_size
                    else model_inputs
                )
                async_output = async_model_executor(*device_inputs)
            async_outputs[model_name].append(async_output)

        # Run collection on-device models (multi-component pipelines).
        for model_name, collection_executor in collection_inference_models.items():
            device_inputs = (
                (x.split(model_batch_size, dim=0) for x in model_inputs)
                if model_batch_size
                else model_inputs
            )
            output_gen = collection_executor.run_model_for_eval(
                device_inputs, model_batch_size
            )
            # Advance past the first yield so the first component's hub job is submitted.
            next(output_gen)
            collection_model_generators[model_name].append(output_gen)

        # Run the remaining local models separately.
        if len(local_inference_models) == 0:
            continue

        # Run local inference on smaller batch size
        local_dataset = DatasetFromIOTuples(model_inputs, ground_truth_values)
        local_dataloader = DataLoader(local_dataset, model_batch_size)
        for sample in tqdm(local_dataloader, disable=IsOnCIEnvvar.get()):
            local_model_inputs, local_ground_truth_values, *_ = sample
            local_model_inputs = _torch_io_to_tuple(local_model_inputs)

            # Run inference on this smaller batch, add to evaluator.
            for model_name, local_model_executor in local_inference_models.items():
                if isinstance(local_model_executor, CollectionAppEvaluateProtocol):
                    batch_output_gen = local_model_executor.run_model_for_eval(
                        local_model_inputs, model_batch_size=1
                    )
                    # Exhaust the generator without retaining intermediate yields.
                    try:
                        while True:
                            next(batch_output_gen)
                    except StopIteration as e:
                        batch_output = e.value
                else:
                    batch_output = local_model_executor(*local_model_inputs)
                evaluators[model_name].add_batch(
                    batch_output,
                    local_ground_truth_values,
                )

        if verbose:
            for model_name in local_inference_models:
                print(
                    f"Cumulative {model_name} accuracy on {cumulative_samples} samples: "
                    f"{evaluators[model_name].formatted_accuracy()}"
                )

    # Drain collection model generators round-robin until all have returned their
    # final output (via StopIteration.value) — a pending AsyncOnDeviceResult, or
    # already-computed values if the pipeline ends on CPU.
    for model_name, gens in collection_model_generators.items():
        pending = set(range(len(gens)))
        while pending:
            for i in list(pending):
                try:
                    next(gens[i])
                except StopIteration as e:  # noqa: PERF203
                    async_outputs[model_name].append(e.value)
                    pending.discard(i)

    # Collect on-device accuracy.
    if async_outputs:
        for model_name, batched_outputs in async_outputs.items():
            for batch_idx, sample in enumerate(dataloader):
                model_inputs, ground_truth_values, *_ = sample
                async_model_output = batched_outputs[batch_idx]
                model_output = (
                    async_model_output.wait()
                    if isinstance(async_model_output, AsyncOnDeviceResult)
                    else async_model_output
                )
                evaluators[model_name].add_batch(model_output, ground_truth_values)

                cumulative_samples = (
                    _count_num_samples(model_output) + batch_idx * batch_size
                )
                if verbose:
                    print(
                        f"Cumulative {model_name} accuracy on {cumulative_samples} samples: "
                        f"{evaluators[model_name].formatted_accuracy()}"
                    )

    if verbose:
        print("\nFinal Accuracy")
        for model_name, evaluator in evaluators.items():
            print(f"{model_name}: {evaluator.formatted_accuracy()}")

    return evaluators


def evaluate_on_dataset(
    dataset_cls: type[BaseDataset],
    evaluator_func: Callable[..., BaseEvaluator],
    model_executors: Mapping[
        str,
        ExecutableModelProtocol | CollectionAppEvaluateProtocol,
    ],
    input_spec: InputSpec,
    samples_per_job: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    use_cache: bool = False,
) -> EvaluateResult:
    """
    Evaluate model accuracy on a dataset both on device and with PyTorch.

    Parameters
    ----------
    dataset_cls
        The dataset class to use for evaluation.
    evaluator_func
        Function that returns a new evaluator instance to use for eval.
    model_executors
        Models or apps to evaluate, keyed by model identifier.
    input_spec
        Desired model input spec. Input types are unused; only shapes and
        order are considered. Always in channel-first format.
    samples_per_job
        Limit on the number of samples to submit in a single inference job.
        If not specified, uses the default value set on the dataset.
    num_samples
        The number of samples to use for evaluation.
        If not set, uses the minimum of the samples_per_job and DEFAULT_NUM_EVAL_SAMPLES.
    seed
        The random seed to use when subsampling the dataset. If not set, creates a deterministic subset.
    use_cache
        If set, will upload the full dataset to hub and store a local copy.
        This prevents re-uploading data to hub for each evaluation, with the
        tradeoff of increased initial overhead.

    Returns
    -------
    result : EvaluateResult
        Contains torch accuracy, quant cpu accuracy, and on device accuracy.
        Quant cpu accuracy is the accuracy from running the quantized ONNX on the CPU.
        If any accuracy was not computed, its value will be None.
    """
    torch_executor_local = model_executors.get("torch")
    on_device_model = model_executors.get("on-device")
    quantized_executor = model_executors.get("quant cpu")

    dataset_name = dataset_cls.dataset_name()
    samples_per_job = samples_per_job or dataset_cls.default_samples_per_job()
    num_samples = num_samples or min(samples_per_job, DEFAULT_NUM_EVAL_SAMPLES)

    if use_cache:
        input_names = list(input_spec)
        channel_last_input = []
        if isinstance(on_device_model, AsyncOnDeviceModel):
            channel_last_input = on_device_model.channel_last_input
            if channel_last_input and (
                torch_executor_local is not None or quantized_executor is not None
            ):
                raise ValueError(
                    "If cache is enabled, a compiled model with channel last input cannot be evaluated alongside a torch / quantized onnx model."
                )

        # When the cache is enabled, we don't want to load the torch dataset unless
        # the cache is out of date. Therefore we pass the dataset class to
        # the function that populates the cache, so it can instantiate the dataset only if necessary.
        _populate_data_cache(
            dataset_cls,
            DatasetSplit.VAL,
            input_spec,
            samples_per_job,
            seed,
            input_names,
            channel_last_input,
        )
        torch_dataset = HubDataset(
            dataset_name,
            num_samples,
            input_names,
            channel_last_input,
            input_spec,
        )
        dataloader = DataLoader(
            torch_dataset, batch_size=samples_per_job, shuffle=False
        )
        num_samples = len(torch_dataset)
    else:
        try:
            source_torch_dataset = instantiate_dataset(
                dataset_cls, DatasetSplit.VAL, input_spec
            )
        except UnfetchableDatasetError as e:
            if e.installation_steps is None:
                raise CLIUserError(
                    f"The chosen dataset for model evaluation ({e.dataset_name}) is not publicly available. If you are running `evaluate.py`, please reach out at https://github.com/quic/ai-hub-models/issues to request accuracy information."
                ) from None
            raise
        if num_samples == -1:
            num_samples = len(source_torch_dataset)
        _validate_inputs(num_samples, source_torch_dataset)
        if seed is not None:
            dataloader = DataLoader(
                dataset=sample_dataset(source_torch_dataset, num_samples, seed),
                batch_size=samples_per_job,
                shuffle=False,
            )
        else:
            dataloader = get_deterministic_sample(
                source_torch_dataset, num_samples, samples_per_job
            )

    actual_samples = len(dataloader.dataset)  # type: ignore[arg-type]
    if actual_samples != num_samples and num_samples not in (-1, actual_samples):
        print(
            f"Evaluating on {actual_samples} samples (requested {num_samples}; rounded up to include complete images)."
        )
    else:
        print(f"Evaluating on {actual_samples} samples.")

    results = evaluate(
        dataloader,
        evaluator_func,
        model_executors=model_executors,
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
    ON_DEVICE = ("on-device", "physical device via AI Hub Workbench (slow)")
    LOCAL_DEVICE = ("local-device", "running on local device like X Elite")

    def __new__(cls, value: str, description: str) -> Self:
        # object.__new__ so we bypass Enum.__init__ machinery
        obj = object.__new__(cls)
        obj._value_ = value  # this is the "real" .value
        obj.description = description  # store your help-text
        return obj

    @staticmethod
    def from_string(string: str) -> EvalMode:
        key = string.replace("-", "_").upper()
        return EvalMode[key]

    def __str__(self) -> str:
        return self.value


def evaluate_session_on_dataset(
    session: onnxruntime.InferenceSession,
    torch_model: WorkbenchModel | CollectionModel,
    dataset_cls: type[BaseDataset],
    num_samples: int | None = None,
) -> tuple[float, str]:
    """
    Evaluate model accuracy on a dataset using ONNX runtime.

    Parameters
    ----------
    session
        ONNX session to evaluate.
    torch_model
        The torch model to evaluate locally to compare accuracy.
    dataset_cls
        The dataset class to use for evaluation.
    num_samples
        The number of samples to use for evaluation.
        If not set, uses the minimum of the samples_per_job and DEFAULT_NUM_EVAL_SAMPLES.

    Returns
    -------
    accuracy : float
        Accuracy score as a float.
    formatted_accuracy : str
        Formatted accuracy as a string.
    """
    assert isinstance(torch_model, WorkbenchModel), (
        "Evaluation is not yet supported for CollectionModels."
    )
    source_torch_dataset = instantiate_dataset(
        dataset_cls, DatasetSplit.VAL, torch_model.get_input_spec()
    )
    num_samples = num_samples or DEFAULT_NUM_EVAL_SAMPLES

    _validate_inputs(num_samples, source_torch_dataset)

    dataloader = get_deterministic_sample(source_torch_dataset, num_samples, None)

    print(f"Evaluating on {num_samples} samples.")
    evaluator = torch_model.get_evaluator()
    inputs, outputs = extract_io_types_from_onnx_model(session)
    session_wrapper = OnnxSessionTorchWrapper(session, inputs, outputs)

    for sample in dataloader:
        model_inputs, ground_truth_values, *_ = sample

        for j, model_input in tqdm(
            enumerate(model_inputs), total=len(model_inputs), disable=IsOnCIEnvvar.get()
        ):
            ground_truth: torch.Tensor | tuple[torch.Tensor, ...]
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
