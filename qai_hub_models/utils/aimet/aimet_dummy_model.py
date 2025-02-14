# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Optional
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from onnx import load_model as load_onnx_model
from onnx import save_model as save_onnx_model
from packaging.version import Version

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

    def __init__(self, model: torch.nn.Module, aimet_encodings: str):
        super().__init__()
        self.model = model
        self.aimet_encodings = aimet_encodings

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
        bundle_external_weights: bool = False,
        output_names: Optional[list[str]] = None,
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
        zip_base_dir = Path(f"{model_name}.aimet")
        zip = self._use_zip_file()

        with ExitStack() as stack:
            if zip:
                # Use temporary directory for preparation
                tmpdir = stack.enter_context(qaihm_temp_dir())
            else:
                tmpdir = output_dir

            base_path = Path(tmpdir) / zip_base_dir
            os.makedirs(base_path, exist_ok=True)

            onnx_file_path = str(base_path / f"{model_name}.onnx")
            encoding_file_path = str(base_path / f"{model_name}.encodings")
            torch_inputs = tuple(make_torch_inputs(input_spec))

            if Version(torch.__version__) < Version("2.4.0"):
                print()
                print(
                    f"WARNING: You are using PyTorch {torch.__version__}, which pre-dates significant ONNX export optimizations"
                )
                print(
                    "         introduced in 2.4.0. We recommend upgrading PyTorch version to speed up this step:"
                )
                print()
                print("         pip install torch==2.4.0")
                print()

            torch.onnx.export(
                self,
                torch_inputs,
                onnx_file_path,
                input_names=[name for name in input_spec],
                output_names=output_names,
                opset_version=17,
            )

            self._adapt_aimet_encodings(
                self.aimet_encodings, encoding_file_path, onnx_file_path
            )

            external_weights_file_path = ""
            if external_weights and zip:
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

            if zip:
                zip_path = os.path.join(output_dir, f"{model_name}.aimet.zip")
                zip_aimet_model(
                    zip_path,
                    zip_base_dir,
                    onnx_file_path,
                    encoding_file_path,
                    external_weights_file_path,
                )
                return zip_path
            else:
                # This path is persistent
                return base_path.as_posix()

        return ""  # mypy requires this for some reason

    def _use_zip_file(self) -> bool:
        """
        Should the return of convert_to_hub_source_model be zipped.
        """
        return True

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Overridable file that adapts the AIMET encodings.
        """
        shutil.copyfile(src=src_encodings_path, dst=dst_encodings_path)
