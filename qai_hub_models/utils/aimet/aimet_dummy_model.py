# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from onnx import load_model as load_onnx_model
from onnx import save_model as save_onnx_model

from qai_hub_models.evaluators.base_evaluators import _DataLoader
from qai_hub_models.models.protocols import (
    PretrainedHubModelProtocol,
    QuantizableModelProtocol,
)
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs


def zip_aimet_model(
    zip_path: str,
    zip_base_dir: Path,
    model_file_path: str,
    encoding_file_path: str,
    external_data_file_path: str = "",
):
    """
    Package and zip .aimet model

    Parameters:
        zip_path: output path for zipped model
        zip_base_dir: base dir path to copy model content
        model_file_path: model file path
        encoding_file_path: encodings file path
        external_data_path: external data file path
    """

    # compresslevel defines how fine compression should run
    # higher the level, heavier algorithm is used leading to more time.
    # For large models, higher compression takes longer time to compress.
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=4) as zip_object:
        zip_object.write(os.path.dirname(model_file_path), zip_base_dir)

        zip_object.write(
            model_file_path,
            os.path.join(zip_base_dir, os.path.basename(model_file_path)),
        )
        if external_data_file_path:
            zip_object.write(
                external_data_file_path,
                os.path.join(zip_base_dir, os.path.basename(external_data_file_path)),
            )
        zip_object.write(
            encoding_file_path,
            os.path.join(zip_base_dir, os.path.basename(encoding_file_path)),
        )


class AimetEncodingLoaderMixin(PretrainedHubModelProtocol, QuantizableModelProtocol):
    """
    Mixin that fakes Torch->ONNX export with encodings similar to AIMET
      - Export Torch model to ONNX and load pre-computed encodings
    """

    def __init__(self, model, aimet_encoding_path: str):
        super().__init__()
        self.model = model
        self.encodings_path = aimet_encoding_path

    def quantize(
        self,
        data: _DataLoader,
        num_samples: int | None = None,
        device: str = "cpu",
        requantize_model_weights=False,
        data_has_gt=False,
    ) -> None:
        raise RuntimeError(
            "AimetEncodingLoaderMixin does not support model quantization."
        )

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        output_names: Optional[List[str]] = None,
    ) -> str:
        """
        Converts the torch module to a zip file containing an
        unquantized ONNX model and an aimet quantization encodings file.
        """
        if model_name is None:
            model_name = self.__class__.__name__
        if not input_spec:
            input_spec = self.get_input_spec()

        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
        zip_base_dir = Path(f"{model_name}.aimet")

        with qaihm_temp_dir() as tmpdir:
            base_path = Path(tmpdir) / zip_base_dir
            if base_path.exists():
                shutil.rmtree(base_path)
            os.makedirs(base_path)

            onnx_file_path = str(base_path / f"{model_name}.onnx")
            encoding_file_path = str(base_path / f"{model_name}.encodings")
            torch.onnx.export(
                self.model,
                tuple(make_torch_inputs(input_spec)),
                onnx_file_path,
                input_names=[name for name in input_spec],
                output_names=output_names,
            )

            shutil.copyfile(self.encodings_path, encoding_file_path)
            external_weights_file_path = ""

            if external_weights:
                external_weights_file_name = f"{model_name}.data"
                external_weights_file_path = str(base_path / external_weights_file_name)
                # Torch exports to onnx with external weights scattered in a directory.
                # Save ONNX model with weights to one file.
                onnx_model = load_onnx_model(onnx_file_path)
                save_onnx_model(
                    onnx_model,
                    onnx_file_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=external_weights_file_name,
                )

            zip_aimet_model(
                zip_path,
                zip_base_dir,
                onnx_file_path,
                encoding_file_path,
                external_weights_file_path,
            )
        return zip_path
