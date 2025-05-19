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
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import onnx
import piqaro
import piqaro.onnx
import torch
from transformers import AutoConfig
from transformers.models.llama import LlamaConfig

from qai_hub_models.models._shared.llama3.export import export_model
from qai_hub_models.models.llama_v3_2_3b_chat import MODEL_ID as model_id_orig
from qai_hub_models.models.llama_v3_2_3b_chat.model import MODEL_ASSET_VERSION, Llama3_2
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.onnx_helpers import (
    torch_onnx_export_with_large_model_size_check,
)
from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

OPT_METHODS = ["no_opt", "manual", "piqaro_torch", "piqaro_onnx"]

MODEL_NAME = "llama_v3_2_3b"

HF_REPO_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def piqaro_onnx_large_model(onnx_model, sample_input, export_dir):
    # Convert to piQaro/PyTorch format
    torch_model = piqaro.onnx._acquire(onnx_model)
    # Without opt(torch_model), got "Exporting the operator 'aten::_to_copy'
    # to ONNX opset version 17 is not supported."

    # Optimize
    opt = piqaro.Optimizer()
    opt(torch_model)

    # Export back to ONNX
    # For models > 2GB, must specify an absolute path so weight files
    # can be written next to the .onnx file
    onnx_path = os.path.join(export_dir, "model.onnx")
    logger.info(f"Saving piqaro-onnx optimized ONNX model to {onnx_path}")

    torch_onnx_export_with_large_model_size_check(torch_model, sample_input, onnx_path)

    onnx_model = onnx.load(onnx_path)
    import onnxsim

    onnx_model, _ = onnxsim.simplify(onnx_model)

    # TODO: rename the first input as input_ids for split_onnx to work correctly
    return onnx_model


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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where ONNX files are stored. Defaults to "
            f"./build/{MODEL_NAME}_<opt>.onnx.zip"
        ),
    )
    parser.add_argument(
        "--truncate-model",
        action="store_true",
        help="True to truncate to a small variant for prototype",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="For reproducibility.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    truncate_model = args.truncate_model

    assert args.opt in OPT_METHODS, f"Unsupported {args.opt}"
    skip_optimizations = None
    if "piqaro" in args.opt or "no_opt" in args.opt:
        skip_optimizations = ["sha_attention", "rank4_rms_norm"]

    trunc_name = "_trunc" if truncate_model else ""
    output_dir = args.output_dir or str(
        Path() / "build" / f"{MODEL_NAME}{trunc_name}_{args.opt}"
    )

    # Use absolute path for onnx.save_model to work properly for >2GB model
    output_dir = Path(output_dir).resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    class Llama3_2_PiQaro(Llama3_2):
        def __init__(self, *args, **kwargs):
            if truncate_model:
                kwargs["load_pretrained"] = False  # Vocab size mismatch
            super().__init__(*args, **kwargs)

        def _llm_config(self, _make_small_for_debugging: bool = False) -> LlamaConfig:
            """
            Construct and return a HuggingFace LLM config.
            """
            if not truncate_model:
                return super()._llm_config(
                    _make_small_for_debugging=_make_small_for_debugging
                )

            llm_config = AutoConfig.from_pretrained(
                HF_REPO_NAME, trust_remote_code=True
            )
            llm_config.num_hidden_layers = 8  # 28 in 3b model
            llm_config.vocab_size = 13  # original: 128256
            llm_config._attn_implementation = "eager"
            llm_config._attn_implementation_internal = "eager"

            # Force use_cache=true for all LLMs
            llm_config.use_cache = True

            return llm_config

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
                logger.info(
                    "Ignoring bundle_external_weights=False and bundle "
                    "the weights into one file to meet Hub requirement"
                )

            dummy_input = tuple(make_torch_inputs(input_spec))
            torch_model = self
            if args.opt == "piqaro_torch":
                logger.info("Optimizing with Piqaro torch")
                torch_model = piqaro.optimize(torch_model, dummy_input)

            # Export to ONNX
            logger.info("Exporting to onnx...")
            onnx_transforms = None
            if args.opt == "piqaro_onnx":
                # Create a temporary directory that will persist until export finishes.
                with tempfile.TemporaryDirectory() as tmpdir:

                    def onnx_transforms(onnx_model):
                        import onnxsim

                        onnx_model, _ = onnxsim.simplify(onnx_model)

                        # TODO: simplify after piqaro fixes
                        # https://github.qualcomm.com/Hexagon-Architecture/piqaro/issues/914
                        # return piqaro.onnx.optimize(onnx_model)

                        return piqaro_onnx_large_model(onnx_model, dummy_input, tmpdir)

                    output_dir = export_torch_to_onnx_zip(
                        torch_model,
                        str(output_dir),
                        dummy_input,
                        input_names=list(input_spec.keys()),
                        output_names=output_names,
                        onnx_transforms=onnx_transforms,
                        skip_zip=True,
                    )
            else:
                """
                # This crashes with piqaro_torch opt
                def onnx_transforms(onnx_model):
                    import onnxsim

                    onnx_model, _ = onnxsim.simplify(onnx_model)
                    return onnx_model
                """
                output_dir = export_torch_to_onnx_zip(
                    torch_model,
                    str(output_dir),
                    dummy_input,
                    input_names=list(input_spec.keys()),
                    output_names=output_names,
                    onnx_transforms=onnx_transforms,
                    skip_zip=True,
                )

            return output_dir

    model_cls = Llama3_2_PiQaro
    model_name = model_id_orig + f"_{args.opt}"

    if truncate_model:
        num_splits = 1
        num_layers_per_split = 2  # doesn't matter
    else:
        num_splits = 3 if args.opt == "manual" else 4
        num_layers_per_split = 14 if args.opt == "manual" else 10
    logger.info(f"Split parameters: {num_splits=}, {num_layers_per_split=}")

    all_components = [f"part_{i + 1}_of_{num_splits}" for i in range(num_splits)]
    all_sub_components = {
        f"part_{i + 1}_of_{num_splits}": [
            f"prompt_{i + 1}_of_{num_splits}",
            f"token_{i + 1}_of_{num_splits}",
        ]
        for i in range(num_splits)
    }

    devices = [
        "Snapdragon X Elite CRD",
        "Samsung Galaxy S23 (Family)",
        "Samsung Galaxy S24 (Family)",
    ]
    for device in devices:
        export_model(
            model_cls=model_cls,
            model_name=model_name,
            model_asset_version=MODEL_ASSET_VERSION,
            components=all_components,
            sub_components=all_sub_components,
            num_layers_per_split=num_layers_per_split,
            output_dir=output_dir,
            _skip_optimizations=skip_optimizations,
            device=device,
            skip_inferencing=True,
            skip_downloading=True,
            model_cache_mode=CacheMode.DISABLE,
        )
