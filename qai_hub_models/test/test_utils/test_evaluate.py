# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable

import pytest
import torch

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import DataLoader, evaluate
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper


class VariableIODummyModel(BaseModel):
    """
    Dummy AI Hub model that allows changing the number of inputs / outputs.
    """

    DEFAULT_IO_SHAPE = (1, 3, 2, 2)

    def __init__(
        self,
        num_inputs: int = 1,
        num_outputs: int = 1,
        shape: tuple[int, ...] = DEFAULT_IO_SHAPE,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.shape = shape
        assert self.num_inputs > 0
        assert self.num_outputs > 0

    @staticmethod
    def from_pretrained(
        num_inputs: int = 1,
        num_outputs: int = 1,
        shape: tuple[int, ...] = DEFAULT_IO_SHAPE,
    ):
        return VariableIODummyModel(num_inputs, num_outputs, shape)

    def forward(
        self, *args: torch.Tensor | int | float, **kwargs: torch.Tensor | int | float
    ) -> torch.Tensor | int | float | tuple[torch.Tensor | int | float, ...]:
        inputs: dict[str, torch.Tensor | int | float] = {}

        for i, arg in enumerate(args):
            inputs[f"in{i}"] = arg

        for kwarg_name, kwarg in kwargs.items():
            assert kwarg_name not in inputs, f"Specified input arg {kwarg_name} twice."
            assert (
                kwarg_name in self.get_input_spec().keys()
            ), f"Unknown input arg {kwarg_name}"
            inputs[kwarg_name] = kwarg

        if len(inputs) < len(self.get_input_spec()):
            raise ValueError(
                f"Missing inputs: {self.get_input_spec().keys() - inputs.keys()}"
            )
        if len(inputs) > len(self.get_input_spec()):
            raise ValueError(
                f"{len(self.get_input_spec())} Provided additional inputs: {inputs.keys() - self.get_input_spec().keys()}"
            )

        out = []
        for outIdx in range(0, self.num_outputs):
            out.append(inputs[f"in{min(self.num_inputs - 1, outIdx)}"] * 2)
        if len(out) < self.num_inputs:
            for i in range(len(out), self.num_inputs):
                out[min(self.num_outputs - 1, i - len(out))] *= inputs[f"in{i}"]

        return tuple(out) if len(out) != 1 else out[0]

    @classmethod
    def get_input_spec(
        cls, num_inputs: int = 1, shape: tuple[int, ...] = DEFAULT_IO_SHAPE
    ) -> InputSpec:
        return {f"in{i}": (shape, "float32") for i in range(0, num_inputs)}

    def _get_input_spec_for_instance(self):
        return self.__class__.get_input_spec(self.num_inputs, self.shape)

    @staticmethod
    def get_output_names(num_outputs: int = 1) -> list[str]:
        return [f"out{i}" for i in range(0, num_outputs)]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.num_outputs)

    @classmethod
    def get_channel_last_inputs(
        cls, num_inputs: int = 1, shape: tuple[int, ...] = DEFAULT_IO_SHAPE
    ) -> list[str]:
        if len(shape) == 4:
            return list(cls.get_input_spec(num_inputs, shape).keys())
        return []

    def _get_channel_last_inputs_for_instance(self) -> list[str]:
        return self.__class__.get_channel_last_inputs(self.num_inputs, self.shape)

    @classmethod
    def get_channel_last_outputs(
        cls, num_outputs: int = 1, shape: tuple[int, ...] = DEFAULT_IO_SHAPE
    ) -> list[str]:
        if len(shape) == 4:
            return cls.get_output_names(num_outputs)
        return []

    def _get_channel_last_outputs_for_instance(self) -> list[str]:
        return self.__class__.get_channel_last_outputs(self.num_outputs, self.shape)


class DummyEvaluator(BaseEvaluator):
    def __init__(self, num_outputs: int):
        self.num_outputs = num_outputs
        self.dummy_metric = 0
        assert self.num_outputs > 0

    @classmethod
    def from_dummy_model(cls, model: VariableIODummyModel) -> DummyEvaluator:
        return cls(model.num_outputs)

    @classmethod
    def get_evaluator_from_model_func(
        cls, model: VariableIODummyModel
    ) -> Callable[[], DummyEvaluator]:
        def get_evaluator() -> DummyEvaluator:
            return cls.from_dummy_model(model)

        return get_evaluator

    def add_batch(
        self,
        output,
        gt,
    ) -> None:
        if self.num_outputs != 1:
            assert isinstance(output, tuple) or isinstance(output, list)
            assert isinstance(gt, tuple) or isinstance(gt, list)
        else:
            output = tuple(
                output,
            )
            gt = tuple(
                gt,
            )

        assert len(output) == len(gt)
        for output, gt in zip(output, gt):
            assert (
                isinstance(output, torch.Tensor)
                or isinstance(output, float)
                or isinstance(output, int)
            )
            assert (
                isinstance(gt, torch.Tensor)
                or isinstance(gt, float)
                or isinstance(gt, int)
            )

        self.dummy_metric = self.dummy_metric + 1

    def reset(self):
        self.dummy_metric = 0

    def get_accuracy_score(self):
        return self.dummy_metric

    def formatted_accuracy(self):
        return f"batch count: {self.dummy_metric}"


class DummyDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit,
        num_inputs: int,
        num_outputs: int,
        shape: tuple[int, ...],
        num_samples: int,
        dtype=torch.float32,
    ):
        super().__init__("", split)
        assert num_samples >= self.default_samples_per_job()
        self.data = [
            tuple(torch.rand(shape, dtype=dtype) for _ in range(0, num_inputs))
            for __ in range(0, num_samples)
        ]
        self.gt = [
            tuple(torch.rand(shape, dtype=dtype) for _ in range(0, num_outputs))
            for __ in range(0, num_samples)
        ]

    @classmethod
    def from_dummy_model(
        cls, model: VariableIODummyModel, split: DatasetSplit, num_samples: int
    ) -> DummyDataset:
        return cls(
            split,
            model.num_inputs,
            model.num_outputs,
            model.shape[1:],  # remove batch dimension, dataloader will handle it
            num_samples,
            torch.float32,
        )

    def __len__(self):
        return len(self.data)

    def _download_data(self):
        return

    def _validate_data(self) -> bool:
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 10

    @classmethod
    def default_num_calibration_samples(cls) -> int:
        return cls.default_samples_per_job()

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]:
        data = self.data[index]
        gt = self.gt[index]
        return data if len(data) != 1 else data[0], gt if len(gt) != 1 else gt[0]


@pytest.mark.parametrize("shuffle", (True, False))
@pytest.mark.parametrize(
    ["num_samples", "dataloader_batch_size", "model_batch_size"],
    (
        (100, 10, 1),
        (100, 10, 5),
        (20, 1, 1),
        (20, 20, 20),
        (100, 18, 16),
        (100, 16, 32),
    ),
)
@pytest.mark.parametrize(
    ["num_inputs", "num_outputs"],
    (
        (1, 1),
        (1, 3),
        (2, 1),
        (3, 7),
        (4, 2),
    ),
)
def test_local_evaluate(
    num_inputs: int,  # number of model inputs
    num_outputs: int,  # number of model output
    num_samples: int,  # number of samples to eval
    dataloader_batch_size: int,  # batch size of data loader passed to evaluate
    model_batch_size: int,  # batch size of compiled ONNX model
    shuffle: bool,  # random dataloader shuffle on/off
):
    """Test local evaluation for PyTorch and ONNX models."""
    model = VariableIODummyModel(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        shape=(model_batch_size, 3, 2, 2),
    )
    with qaihm_temp_dir() as tmpdir:
        onnx_path = f"{tmpdir}/model.onnx"
        torch.onnx.export(
            model, tuple(make_torch_inputs(model.get_input_spec())), onnx_path
        )
        onnx_model = OnnxModelTorchWrapper.OnCPU(onnx_path)

    evaluator_func = DummyEvaluator.get_evaluator_from_model_func(model)
    dataset = DummyDataset.from_dummy_model(model, DatasetSplit.VAL, num_samples)
    dataloader = DataLoader(dataset, dataloader_batch_size, shuffle)

    if dataloader_batch_size % model_batch_size != 0:
        with pytest.raises(
            ValueError, match=".*must evenly divide the DataLoader's batch size.*"
        ):
            out = evaluate(
                dataloader,
                evaluator_func,
                {"torch": model, "onnx": onnx_model},
                model_batch_size,
                verbose=False,
            )
    else:
        out = evaluate(
            dataloader,
            evaluator_func,
            {"torch": model, "onnx": onnx_model},
            model_batch_size,
            verbose=False,
        )
        assert out["torch"].get_accuracy_score() == num_samples // model_batch_size
        assert out["onnx"].get_accuracy_score() == num_samples // model_batch_size
