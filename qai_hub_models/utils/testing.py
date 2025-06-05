# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import qai_hub as hub
from qai_hub.client import SourceModelType
from tabulate import tabulate

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.asset_loaders import (
    always_answer_prompts,
    load_yaml,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    BasePrecompiledModel,
    CollectionModel,
)
from qai_hub_models.utils.evaluate import (
    CACHE_SAMPLES_PER_JOB_FILE,
    get_dataset_path,
    get_torch_val_dataloader,
)
from qai_hub_models.utils.inference import (
    AsyncOnDeviceResult,
    dataset_entries_from_batch,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.quantization import get_calibration_data
from qai_hub_models.utils.testing_async_utils import (  # noqa: F401
    append_line_to_file,
    get_artifact_filepath,
    get_artifacts_dir,
    get_artifacts_dir_opt,
    get_compile_job_ids_file,
    get_dataset_ids_file,
    get_inference_job_ids_file,
    get_profile_job_ids_file,
    get_quantize_job_ids_file,
    is_hub_testing_async,
)


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


def mock_tabulate_fn(df: pd.DataFrame, **kwargs) -> tuple[list[str], str]:
    psnr_values = []
    for i, (_, value) in enumerate(df.iterrows()):
        psnr_values.append(value.psnr)
    return psnr_values, tabulate(df, **kwargs)  # pyright: ignore[reportArgumentType]


def get_and_sync_datasets_cache_dir(
    has_channel_transpose: bool,
    dataset_name: str,
    samples_per_job: int,
    model_cls: type[BaseModel] | type[CollectionModel],
) -> Path:
    folder_name = "hub_datasets"
    if not has_channel_transpose:
        folder_name += "_nt"
    dir_path = get_artifacts_dir() / folder_name / dataset_name
    if dir_path.exists():
        return dir_path
    with qaihm_temp_dir() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_ids_filepath = get_dataset_ids_file()
        dataset_ids = load_yaml(dataset_ids_filepath)
        input_key, gt_key = get_val_dataset_id_keys(dataset_name, has_channel_transpose)
        # In most cases, the input and gt validation data have been created
        # and stored in the dataset_ids yaml. In the rare case it isn't, do so here.
        if input_key not in dataset_ids or gt_key not in dataset_ids:
            get_hub_val_dataset(
                dataset_name,
                dataset_ids_filepath,
                model_cls,
                has_channel_transpose,
                samples_per_job,
            )
            dataset_ids = load_yaml(dataset_ids_filepath)
        with open(tmp_path / f"samples_per_job_{samples_per_job}.txt", "w") as f:
            f.write(dataset_ids[input_key] + " " + dataset_ids[gt_key])

        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        hub.get_dataset(dataset_ids[input_key]).download(
            str(get_dataset_path(cache_path, dataset_ids[input_key]))
        )
        hub.get_dataset(dataset_ids[gt_key]).download(
            str(get_dataset_path(cache_path, dataset_ids[gt_key]))
        )
        with open(cache_path / CACHE_SAMPLES_PER_JOB_FILE, "w") as f:
            f.write(str(samples_per_job) + "\n")

        dir_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_path, dir_path)
    return dir_path


def mock_get_calibration_data(
    model: BaseModel, input_spec: InputSpec, num_samples: int
) -> hub.Dataset:
    cache_prefix = model.calibration_dataset_name() or model.__class__.__name__
    cache_key = cache_prefix + "_train"
    dataset_ids_file = get_dataset_ids_file()
    dataset_ids = load_yaml(dataset_ids_file)
    if dataset_ids and cache_key in dataset_ids:
        return hub.get_dataset(dataset_ids[cache_key])
    dataset = get_calibration_data(model, input_spec, num_samples)
    hub_dataset = hub.upload_dataset(dataset)
    append_line_to_file(dataset_ids_file, f"{cache_key}: {hub_dataset.dataset_id}")
    return hub_dataset


def get_val_dataset_id_keys(
    dataset_name: str, apply_channel_transpose: bool
) -> tuple[str, str]:
    base_name = f"{dataset_name}_val_"
    if not apply_channel_transpose:
        base_name += "nt_"  # no transpose
    return base_name + "input", base_name + "gt"


def get_hub_val_dataset(
    dataset_name: str,
    ids_file: Path,
    model_cls: type[BaseModel] | type[CollectionModel],
    apply_channel_transpose: bool,
    num_samples: int | None = None,
) -> hub.Dataset:
    """
    Creates a hub dataset with a chunk of validation data from the given dataset.

    If the dataset ids are already present in the file, the function
    just loads and returns it.

    Otherwise, it creates the dataset and writes the ids to the file.

    The dataset is sampled by taking every N samples for the largest possible N
    that produces the number of requested samples.

    The dataset is produced using the input spec and channel last inputs of a
    representative model. The assumption being that all models using this dataset
    have the same values for both of those things.

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
    assert issubclass(
        model_cls, BaseModel
    ), "CollectionModel is not yet supported by this function."
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


@contextlib.contextmanager
def patch_qai_hub(model_type: SourceModelType = SourceModelType.ONNX):
    """
    Generic AI Hub patch with assertable mocks. Example::

        with patch_qai_hub() as mock_hub:
            hub.submit_profile_job(...)
            mock_hub.submit_profile_job.assert_called()
    """
    # Model
    model = mock.Mock(hub.Model)
    model.model_id = "mabcd1234"
    model.model_type = model_type

    def _store_device(mock_obj, *args, **kwargs):
        if "device" in kwargs:
            mock_obj.device = kwargs["device"]
        return mock_obj

    # Compile
    mock_compile = mock.Mock(hub.CompileJob)
    mock_compile.job_id = "jcbcd1234"
    mock_compile.get_target_model.return_value = model
    mock_submit_compile_job = mock.Mock()
    mock_submit_compile_job.return_value = mock_compile

    # Profile
    mock_profile = mock.Mock(hub.ProfileJob)
    mock_profile.name = "Profile job"
    mock_profile.job_id = "jpbcd1234"
    mock_profile.model = model
    # heavily abridged profile
    mock_profile.download_profile.return_value = {
        "execution_detail": {},
        "execution_summary": {
            "inference_memory_peak_range": [50000, 100000],
            "estimated_inference_time": 10000,
            "peak_memory_bytes": 12000,
        },
    }
    mock_submit_profile_job = mock.Mock()
    mock_submit_profile_job.return_value = mock_profile
    mock_submit_profile_job.side_effect = partial(_store_device, mock_profile)

    # Inference
    mock_inference = mock.Mock(hub.InferenceJob)
    mock_inference.name = "Inference job"
    mock_inference.job_id = "jibcd1234"
    mock_inference.download_output_data.return_value = {
        "output0": [np.array([1.0, 2.0])],
        "output1": [np.array([3.0])],
    }
    mock_submit_inference_job = mock.Mock()
    mock_submit_inference_job.return_value = mock_inference

    # Link
    mock_link = mock.Mock(hub.LinkJob)
    mock_link.name = "Link job"
    mock_link.job_id = "jlbcd1234"
    mock_link.get_target_model.return_value = model
    mock_submit_link_job = mock.Mock()
    mock_submit_link_job.return_value = mock_link

    # Quantize
    mock_quantize = mock.Mock(hub.QuantizeJob)
    mock_quantize.name = "Quantize job"
    mock_quantize.job_id = "jqbcd1234"
    mock_quantize.get_target_model.return_value = model
    mock_submit_quantize_job = mock.Mock()
    mock_submit_quantize_job.return_value = mock_quantize

    patch_hub_compile = mock.patch(
        "qai_hub.submit_compile_job", mock_submit_compile_job
    )
    patch_hub_profile = mock.patch(
        "qai_hub.submit_profile_job", mock_submit_profile_job
    )
    patch_hub_inference = mock.patch(
        "qai_hub.submit_inference_job", mock_submit_inference_job
    )
    patch_hub_link = mock.patch("qai_hub.submit_link_job", mock_submit_link_job)
    patch_hub_quantize = mock.patch(
        "qai_hub.submit_quantize_job", mock_submit_quantize_job
    )

    with (
        patch_hub_compile
    ), patch_hub_profile, patch_hub_inference, patch_hub_link, patch_hub_quantize:
        # Yield mocks to allow assertions
        yield SimpleNamespace(
            submit_compile_job=mock_submit_compile_job,
            submit_profile_job=mock_submit_profile_job,
            submit_inference_job=mock_submit_inference_job,
            submit_link_job=mock_submit_link_job,
            submit_quantize_job=mock_submit_quantize_job,
        )


def has_get_unsupported_reason(cls: type, stop_at_classes: list[type]) -> bool:
    """
    Check whether the 'get_unsupported_reason' attribute is defined in the given class
    or any of its parent classes up to (but not including) any class in stop_at_classes.

    Parameters:
        cls (type): The class to check.
        stop_at_classes (list[type]): A list of classes at which to stop the search in the MRO.

    Returns:
        bool: True if 'get_unsupported_reason' is found in cls or one of its parent classes
              before reaching any of the stop_at_classes; False otherwise.
    """
    for base in cls.__mro__:
        if base in stop_at_classes:
            break
        if "get_unsupported_reason" in base.__dict__:
            return True
    return False


def _skip_if_unsupported_reason(
    model_cls: type[BaseModel] | type[BasePrecompiledModel],
    runtime: TargetRuntime,
    device: ScorecardDevice,
):
    if not has_get_unsupported_reason(model_cls, [BaseModel, BasePrecompiledModel]):
        return
    # check get_unsupported_reason
    if issubclass(model_cls, BaseModel):
        model = model_cls.from_pretrained()
    else:
        model = model_cls.from_precompiled()  # type: ignore
    hub_device = device.execution_device
    reason = model.get_unsupported_reason(runtime, hub_device)  # type: ignore
    if reason:
        pytest.xfail(reason)


def skip_invalid_runtime_device(
    model_cls: type[BaseModel] | type[BasePrecompiledModel] | type[CollectionModel],
    runtime: TargetRuntime,
    device: ScorecardDevice,
) -> None:
    if issubclass(model_cls, CollectionModel):
        for component_cls in model_cls.component_classes:
            _skip_if_unsupported_reason(component_cls, runtime, device)
        return
    # BaseModel or BasePrecompiledModel
    _skip_if_unsupported_reason(model_cls, runtime, device)
