# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Items defined in this file require that AIMET-ONNX be installed.
"""

from __future__ import annotations

try:
    import onnxruntime as ort
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx

except (ImportError, ModuleNotFoundError):
    raise NotImplementedError(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

import os
from collections.abc import Collection
from pathlib import Path

import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from qai_hub_models.evaluators.base_evaluators import _DataLoader
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.protocols import PretrainedHubModelProtocol
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import Precision, TargetRuntime
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx_helpers import mock_torch_onnx_inference


class AIMETOnnxQuantizableMixin(PretrainedHubModelProtocol):
    """
    It follows closely AIMETQuantizableMixin. The main differences are
    following simplifications:

    - quantize() takes just data and num_samples. (No device,
            requantize_model_weights, data_has_gt)

    - convert_to_onnx_and_aimet_encodings() takes just output_dir and
    model_name. No input_spec, external_weights,
    output_names.

    Due to these simplification, we no longer satisfy QuantizableModelProtocol
    """

    def __init__(
        self,
        quant_sim: QuantSimOnnx,
    ):
        self.quant_sim = quant_sim
        self.input_names = [input.name for input in self.quant_sim.session.get_inputs()]
        self.output_names = [
            output.name for output in self.quant_sim.session.get_outputs()
        ]

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        """
        Same as AIMETQuantizableMixin.get_calibration_data with the addition
        of `num_samples`

        Args:

        num_samples: None to use all. Specify `num_samples` to use fewer. If
        `num_samples` are more than available, use all available (same
        behavior as None)

        """
        return None

    def quantize(
        self,
        data: _DataLoader | None = None,
        num_samples: int | None = None,
    ) -> None:
        """
        Args:

        - data: If None, create data loader from get_calibration_data(), which
        must be implemented.
        """
        if data is None:
            calib_data = self.get_calibration_data(
                target_runtime=TargetRuntime.PRECOMPILED_QNN_ONNX
            )
            if calib_data is None:
                raise ValueError(
                    "`data` must be specified if "
                    "get_calibration_data is not defined."
                )
            data = dataset_entries_to_dataloader(calib_data)

        def _forward(session, _):
            run_onnx_inference(session, data, num_samples)

        print(f"Start QuantSim calibration for {self.__class__.__name__}")
        self.quant_sim.compute_encodings(_forward, tuple())

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        data = self.get_calibration_data(target_runtime=TargetRuntime.QNN)
        assert isinstance(data, dict)
        return data

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """
        QuantSim forward pass with torch.Tensor
        """
        return mock_torch_onnx_inference(self.quant_sim.session, *args, **kwargs)

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
    ) -> str:
        """
        Converts the torch module to a zip file containing an
        unquantized ONNX model and an aimet quantization encodings file.
        """
        if model_name is None:
            model_name = self.__class__.__name__

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        base_dir = Path(f"{model_name}.aimet")

        print(f"Exporting quantized {self.__class__.__name__} to {zip_path}")
        with qaihm_temp_dir() as tmpdir:
            base_path = Path(tmpdir) / base_dir
            os.makedirs(base_path)

            self.quant_sim.export(str(base_path), "model")

            onnx_file_path = str(base_path / "model.onnx")
            encoding_file_path = str(base_path / "model.encodings")
            external_data_file_path = str(base_path / "model.onnx.data")

            if not os.path.exists(external_data_file_path):
                external_data_file_path = ""

            zip_aimet_model(
                zip_path,
                base_dir,
                onnx_file_path,
                encoding_file_path,
                external_data_file_path,
            )

        return zip_path

    def get_hub_quantize_options(self, precision: Precision) -> str:
        """
        AI Hub quantize options recommended for the model.
        """
        return ""


def run_onnx_inference(
    session: ort.InferenceSession,
    dataloader: DataLoader,
    num_samples: int | None = None,
) -> None:
    input_names = [input.name for input in session.get_inputs()]

    total = num_samples or len(dataloader)
    assert dataloader.batch_size is not None
    for i, batch in tqdm(enumerate(dataloader), total=total):
        if num_samples and i * dataloader.batch_size >= num_samples:
            break

        if isinstance(batch, torch.Tensor):
            batch = (batch,)

        if len(batch) != len(input_names):
            raise ValueError(
                "The number of inputs from DataLoader does not match the ONNX model's inputs."
            )

        inputs = {
            input_name: batch[i].numpy() for i, input_name in enumerate(input_names)
        }

        session.run(None, inputs)
