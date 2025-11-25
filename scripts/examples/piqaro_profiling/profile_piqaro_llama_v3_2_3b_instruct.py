# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
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
from pathlib import Path

import piqaro
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llama3.model import Llama3Base
from qai_hub_models.models._shared.llm.export import export_model
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    LLMInstantiationType,
    MainLLMInputType,
)
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.llama_v3_2_3b_instruct import MODEL_ID
from qai_hub_models.models.llama_v3_2_3b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MODEL_ASSET_VERSION,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    Llama3_2_3B,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

OPT_METHODS = ["no_opt", "manual", "piqaro"]

MODEL_NAME = "llama_v3_2_3b"

NUM_LAYERS_TRUNC = 2


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        type=str,
        default="manual",
        help=f"Optimization method. One of {OPT_METHODS}. Default is no_opt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            f"Directory where ONNX files are stored. Defaults to ./build/{MODEL_NAME}_<opt>.onnx.zip"
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
    if args.opt == "piqaro" or "no_opt" in args.opt:
        skip_optimizations = ["sha_attention", "rank4_rms_norm"]

    compile_options = "--qnn_bin_conversion_via_model_library"
    compile_options += " --qairt_version 2.38"
    if args.opt == "piqaro":
        compile_options += " --enable_piqaro"

    link_options = " --qairt_version 2.38"
    profile_options = " --qairt_version 2.38"

    trunc_name = "_trunc" if truncate_model else ""
    output_dir = args.output_dir or str(
        Path() / "build" / f"{MODEL_NAME}{trunc_name}_{args.opt}"
    )
    opt = args.opt
    # Use absolute path for onnx.save_model to work properly for >2GB model
    output_dir = Path(output_dir).resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    class Llama3_2_PiQaro_FP(Llama3Base):
        def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
            if truncate_model:
                llm_config.num_hidden_layers = NUM_LAYERS_TRUNC
                llm_config.vocab_size = 13  # original: 128256
            return llm_config

        @classmethod
        def from_pretrained(
            cls,
            sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
            context_length: int = DEFAULT_CONTEXT_LENGTH,
            **kwargs,  # ignore precision and other args
        ) -> Llama3_2_3B:
            return cls(
                checkpoint=HF_REPO_NAME,
                sequence_length=sequence_length,
                context_length=context_length,
                host_device="cpu",
                _skip_optimizations=skip_optimizations,
                load_pretrained=False,
            )

        @staticmethod
        def get_input_spec(
            llm_config: dict,
            sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
            context_length: int = DEFAULT_CONTEXT_LENGTH,
            main_input_name: str = MainLLMInputType.input_ids.name,
        ) -> InputSpec:
            num_layers = NUM_LAYERS_TRUNC if truncate_model else NUM_LAYERS
            return LLMBase._get_input_spec(
                num_hidden_layers=num_layers,
                sequence_length=sequence_length,
                context_length=context_length,
                hidden_size=HIDDEN_SIZE,
                num_key_value_heads=NUM_KEY_VALUE_HEADS,
                num_attention_heads=NUM_ATTN_HEADS,
                main_input_name=main_input_name,
            )

        def convert_to_hub_source_model(
            self,
            target_runtime: TargetRuntime,
            output_path: str | Path,
            input_spec: InputSpec | None = None,
            check_trace: bool = True,
            external_onnx_weights: bool = False,
            output_names: list[str] | None = None,
        ) -> str | None:
            """Convert to a AI Hub source model appropriate for the export method."""

            def apply_piqaro_onnx(onnx_model):
                import onnxsim

                onnx_model, _ = onnxsim.simplify(onnx_model)
                return piqaro.onnx.optimize(onnx_model)

            onnx_transforms = apply_piqaro_onnx if args.opt == "piqaro" else None
            dummy_input = tuple(make_torch_inputs(input_spec))
            # Need to export to {output_path}/model.onnx
            output_dir = Path(output_path) / "llama_two_layer.onnx"
            output_dir.mkdir(parents=True, exist_ok=True)
            _ = export_torch_to_onnx_zip(
                self,
                output_dir,
                dummy_input,
                input_names=list(input_spec.keys()),
                output_names=output_names,
                onnx_transforms=onnx_transforms,
                skip_zip=True,
            )

            return str(output_dir)

        @staticmethod
        def get_output_names() -> list[str]:
            num_hidden_layers = NUM_LAYERS_TRUNC if truncate_model else NUM_LAYERS
            output_names = ["logits"]
            for layer in range(num_hidden_layers):
                output_names.append(f"past_key_{layer}_out")
                output_names.append(f"past_value_{layer}_out")
            return output_names

        def get_qairt_context_graph_name(
            self, split_index: int, num_splits: int
        ) -> str:
            """
            Get the name of the QAIRT Context Graph applicable for the given sub-component.

            Sequence length (ar...) and context length (cl...) in graph name
            are semantically important to Genie
            """
            if self.sequence_length == 1:
                instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
            else:
                instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
            return f"{instantiation_type.value}_ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"

    model_cls = Llama3_2_PiQaro_FP
    model_name = MODEL_ID + f"_{args.opt}"

    if truncate_model:
        num_splits = 1
        num_layers_per_split = 2  # doesn't matter
    else:
        num_splits = 3 if args.opt == "manual" else 4
        num_layers_per_split = 14 if args.opt == "manual" else 10
    logger.info(f"Split parameters: {num_splits=}, {num_layers_per_split=}")

    devices = [
        # "Snapdragon X Elite CRD",
        # "Samsung Galaxy S23 (Family)",
        "Samsung Galaxy S24 (Family)",
    ]

    onnx_export_dir = (
        Path(__file__).resolve().parent / "build" / "llama_3_2_3b_2layer" / args.opt
    )

    for i, device in enumerate(devices):
        # Disable for the first device, reuse for all other devices.
        cache = CacheMode.DISABLE if i == 0 else CacheMode.ENABLE
        export_model(
            model_cls=model_cls,
            model_name=model_name,
            model_asset_version=MODEL_ASSET_VERSION,
            num_splits=num_splits,
            num_layers_per_split=num_layers_per_split,
            precision=Precision.float,
            device=device,
            output_dir=output_dir,
            _skip_optimizations=skip_optimizations,
            skip_inferencing=True,
            skip_downloading=True,
            model_cache_mode=cache,
            compile_options=compile_options,
            link_options=link_options,
            profile_options=profile_options,
            onnx_export_dir=output_dir,
        )
