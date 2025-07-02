# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Items defined in this file require that AIMET-ONNX be installed.
"""

from __future__ import annotations

try:
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
    from aimet_onnx.sequential_mse.seq_mse import SeqMseParams, SequentialMse
except (ImportError, ModuleNotFoundError):
    pass
import gc
import os
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from qai_hub_models.evaluators.base_evaluators import _DataLoader
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.protocols import PretrainedHubModelProtocol
from qai_hub_models.utils.aimet.aimet_dummy_model import zip_aimet_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx_helpers import mock_torch_onnx_inference
from qai_hub_models.utils.onnx_torch_wrapper import OnnxSessionTorchWrapper


class AIMETOnnxQuantizableMixin(PretrainedHubModelProtocol):
    """
    Mixin that allows a model to be quantized & exported to disk using AIMET.
    Inheritor must implement BaseModel for this mixin to function.
    """

    # For pre-calibrated asset lookup
    model_id: str = ""
    model_asset_version: int = -1

    def __init__(
        self,
        quant_sim: QuantSimOnnx | None,
    ):
        self.quant_sim = quant_sim
        if self.quant_sim is not None:
            self.input_names = [
                input.name for input in self.quant_sim.session.get_inputs()
            ]
            self.output_names = [
                output.name for output in self.quant_sim.session.get_outputs()
            ]

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        # This must be defined by the PretrainedHubModelProtocol ABC
        raise ValueError(
            f"Cannot call convert_to_torchscript on {self.__class__.__name__}"
        )

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        """
        Args:

        num_samples: None to use all. Specify `num_samples` to use fewer. If
        `num_samples` are more than available, use all available (same
        behavior as None)

        """
        return None

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        """
        Returns .onnx and .encodings paths
        """
        if not cls.model_id or cls.model_asset_version == -1:
            raise ValueError("model_id and model_asset_version must be defined")

        subfolder = Path(getattr(cls, "default_subfolder", ""))

        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.onnx"),
        ).fetch()
        try:
            _ = CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                str(subfolder / "model.data"),
            ).fetch()
        except Exception:
            pass  # ignore. No external weight.
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            cls.model_id,
            cls.model_asset_version,
            str(subfolder / "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def _apply_seq_mse(self, data: _DataLoader, num_batches: int):
        assert self.quant_sim is not None
        params = SeqMseParams(num_batches=num_batches)
        seq_mse = SequentialMse(self.quant_sim.model, self.quant_sim, params, data)
        seq_mse.apply_seq_mse_algo()

    def _apply_calibration(self, data: DataLoader, num_batches: int):
        assert self.quant_sim is not None

        def _forward(session, _):
            wrapper = OnnxSessionTorchWrapper(session, quantize_io=False)
            assert data.batch_size is not None
            for i, batch in tqdm(enumerate(data), total=num_batches):
                if num_batches and i * data.batch_size >= num_batches:
                    break

                if isinstance(batch, torch.Tensor):
                    batch = (batch,)

                wrapper.forward(*batch)

                gc.collect()
                torch.cuda.empty_cache()

        self.quant_sim.compute_encodings(_forward, tuple())

    def quantize(
        self,
        data: _DataLoader | None = None,
        num_samples: int | None = None,
        use_seq_mse: bool = False,
    ) -> None:
        """
        Args:

        - data: If None, create data loader from get_calibration_data(), which
        must be implemented.
        """
        if data is None:
            calib_data = self.get_calibration_data()
            if calib_data is None:
                raise ValueError(
                    "`data` must be specified if "
                    "get_calibration_data is not defined."
                )
            data = dataset_entries_to_dataloader(calib_data)

        num_iterations = num_samples or len(data)

        if use_seq_mse:
            self._apply_seq_mse(data=data, num_batches=num_iterations)

        print(f"Start QuantSim calibration for {self.__class__.__name__}")
        self._apply_calibration(data=data, num_batches=num_iterations)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        data = self.get_calibration_data()
        if data is None:
            # Fallback to BaseModel's impl
            data = super()._sample_inputs_impl(input_spec)  # type: ignore
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
        assert self.quant_sim is not None
        return mock_torch_onnx_inference(self.quant_sim.session, *args, **kwargs)

    def save_calibrated_checkpoint(self, output_checkpoint: str) -> None:
        """
        Save AIMET-ONNX checkpoint to output_checkpoint/subfolder, if
        """
        default_subfolder = getattr(self.__class__, "default_subfolder", "")
        export_dir = output_checkpoint
        if default_subfolder:
            export_dir = str(Path(output_checkpoint) / default_subfolder)

        shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        print(f"Saving quantized {self.__class__.__name__} to {export_dir}")
        assert self.quant_sim is not None
        self.quant_sim.export(str(export_dir), "model")
        print(f"{self.__class__.__name__} saved to {export_dir}")

    def get_ort_providers(
        device: torch.device,
    ) -> list[str | tuple[str, dict[str, int]]]:
        if device.type == "cuda":
            return (
                [
                    ("CUDAExecutionProvider", {"device_id": device.index}),
                    "CPUExecutionProvider",
                ]
                if device.index is not None
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        return ["CPUExecutionProvider"]

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        model_name: str | None = None,
        return_zip: bool = True,
    ) -> str:
        """
        Converts the torch module to a zip file containing an unquantized ONNX model
        and an AIMET quantization encodings file if return_zip is True (default).

        If return_zip is False, the model is exported to a directory.
        In that case, the output directory is set to:

            Path(output_dir) / f"{model_name}.aimet"

        and the existing directory is forcefully removed.
        """
        if model_name is None:
            model_name = self.__class__.__name__

        output_dir = Path(output_dir)

        if return_zip:
            # Ensure output_dir exists and define the zip path.
            os.makedirs(output_dir, exist_ok=True)
            zip_path = output_dir / f"{model_name}.aimet.zip"
            base_dir = Path(f"{model_name}.aimet")

            print(f"Exporting quantized {self.__class__.__name__} to {zip_path}")
            # Use a temporary directory to export the model before zipping.
            with qaihm_temp_dir() as tmpdir:
                export_dir = Path(tmpdir) / base_dir
                os.makedirs(export_dir)
                assert self.quant_sim is not None
                self.quant_sim.export(str(export_dir), "model")

                onnx_file_path = str(export_dir / "model.onnx")
                encoding_file_path = str(export_dir / "model.encodings")

                # Attempt to locate external data file.
                # aimet-onnx<=2.0.0 export external data with model.onnx.data
                # aimet-onnx>=2.3.0 export external data with model.data
                # version between 2.0 - 2.3 are broken on large models
                external_data_file_path = ""
                external_data_file_path2 = export_dir / "model.onnx.data"
                external_data_file_path1 = export_dir / "model.data"
                if external_data_file_path1.exists():
                    external_data_file_path = str(external_data_file_path1)
                elif external_data_file_path2.exists():
                    external_data_file_path = str(external_data_file_path2)

                zip_aimet_model(
                    str(zip_path),
                    base_dir,
                    onnx_file_path,
                    encoding_file_path,
                    external_data_file_path,
                )
            return str(zip_path)
        else:
            # Export directly to a directory at output_dir / f"{model_name}.aimet"
            export_dir = output_dir / f"{model_name}.aimet"
            shutil.rmtree(export_dir, ignore_errors=True)
            os.makedirs(export_dir, exist_ok=True)

            print(
                f"Exporting quantized {self.__class__.__name__} to directory {export_dir}"
            )
            assert self.quant_sim is not None
            self.quant_sim.export(str(export_dir), "model")
            return str(export_dir)

    def get_hub_quantize_options(self, precision: Precision) -> str:
        """
        AI Hub quantize options recommended for the model.
        """
        return ""
