# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to profile PiQaro's optimization on
Llama3.2-3b-chat

Install piqaro from https://github.qualcomm.com/Hexagon-Architecture/piqaro
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import piqaro
import piqaro.onnx
import torch

from qai_hub_models.models._shared.llama3.export import export_model
from qai_hub_models.models.llama_v3_2_3b_chat_quantized import MODEL_ID as model_id_orig
from qai_hub_models.models.llama_v3_2_3b_chat_quantized.model import (
    MODEL_ASSET_VERSION,
    Llama3_2_Quantized,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip

OPT_METHODS = ["no_opt", "manual", "piqaro_torch", "piqaro_onnx"]

MODEL_NAME = "llama_v3_2_3b"


class WrapperModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        type=str,
        default="manual",
        help="Optimization method. One of {OPT_METHODS}. Default is no_opt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Snapdragon X Elite CRD",
        help="Hub device",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where ONNX files are stored. Defaults to "
            f"./build/{MODEL_NAME}_<opt>.onnx.zip"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    assert args.opt in OPT_METHODS, f"Unsupported {args.opt}"
    skip_optimizations = None
    if "piqaro" in args.opt or "no_opt" in args.opt:
        skip_optimizations = ["sha_attention", "rank4_rms_norm"]

    output_dir = args.output_dir or str(Path() / "build" / f"{MODEL_NAME}_{args.opt}")
    os.makedirs(output_dir, exist_ok=True)

    class Llama3_2_Quantized_PiQaro(Llama3_2_Quantized):
        # Override the export
        def convert_to_onnx_and_aimet_encodings(
            self,
            output_dir: str | Path,
            input_spec: InputSpec | None = None,
            model_name: str | None = None,
            external_weights: bool = False,
            bundle_external_weights: bool = False,
            output_names: Optional[list[str]] = None,
        ) -> str:
            model_name = model_name or self.__class__.__name__
            input_spec = input_spec or self.get_input_spec()
            assert external_weights
            # Don't zip
            assert not self._use_zip_file()
            output_dir = Path(output_dir)

            if not bundle_external_weights:
                print(
                    "Ignoring bundle_external_weights=False and bundle "
                    "the weights into one file to meet Hub requirement"
                )

            dummy_input = tuple(make_torch_inputs(input_spec))
            torch_model = self
            if args.opt == "piqaro_torch":
                print("Optimizing with Piqaro torch")
                torch_model = piqaro.optimize(torch_model, dummy_input)

            # Export to ONNX
            print("Exporting to onnx...")
            onnx_transforms = None
            if args.opt == "piqaro_onnx":

                def onnx_transforms(onnx_model):
                    import onnxsim

                    onnx_model, _ = onnxsim.simplify(onnx_model)
                    return piqaro.onnx.optimize(onnx_model)

            export_torch_to_onnx_zip(
                torch_model,
                str(output_dir),
                dummy_input,
                input_names=list(input_spec.keys()),
                output_names=output_names,
                onnx_transforms=onnx_transforms,
                skip_zip=True,
            )

            # create an empty encoding
            encoding_file_path = str(output_dir / "model.encodings")
            # Create an empty file at the encoding file path
            with open(encoding_file_path, "w"):
                pass

            return output_dir.as_posix()

    if "piqaro" in args.opt:
        model_cls = Llama3_2_Quantized_PiQaro
    else:
        model_cls = Llama3_2_Quantized
    model_name = model_id_orig + f"_{args.opt}"

    num_splits = 3 if args.opt == "manual" else 4
    num_layers_per_split = 14 if args.opt == "manual" else 10
    all_components = [f"part_{i + 1}_of_{num_splits}" for i in range(num_splits)]
    all_sub_components = {
        f"part_{i + 1}_of_{num_splits}": [
            f"prompt_{i + 1}_of_{num_splits}",
            f"token_{i + 1}_of_{num_splits}",
        ]
        for i in range(num_splits)
    }

    export_model(
        model_cls=model_cls,
        model_name=model_name,
        model_asset_version=MODEL_ASSET_VERSION,
        components=all_components,
        sub_components=all_sub_components,
        num_layers_per_split=num_layers_per_split,
        output_dir=output_dir,
        _skip_optimizations=skip_optimizations,
        device=args.device,
    )
