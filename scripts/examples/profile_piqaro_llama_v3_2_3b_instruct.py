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
from pathlib import Path

import piqaro
import torch
from transformers import PretrainedConfig

from qai_hub_models.models._shared.llama3.export import export_model
from qai_hub_models.models._shared.llama3.model import Llama3Base
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

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

OPT_METHODS = ["no_opt", "manual", "piqaro_torch", "piqaro_onnx"]

MODEL_NAME = "llama_v3_2_3b"

HF_REPO_NAME = "meta-llama/Llama-3.2-3B-Instruct"

NUM_LAYERS_TRUNC = 1


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
        ) -> Llama3_2_3B:
            hub_model_fp = cls(
                min_memory_recommended=0,
                checkpoint=HF_REPO_NAME,
                sequence_length=sequence_length,
                context_length=context_length,
                host_device="cpu",
                _skip_optimizations=skip_optimizations,
                load_pretrained=False,
            )
            if opt == "piqaro_torch":
                dummy_input = tuple(make_torch_inputs(hub_model_fp.get_input_spec()))
                optimized_fx_graph = piqaro.optimize(hub_model_fp, dummy_input)
                hub_model_fp.model = optimized_fx_graph
            return hub_model_fp

        def convert_to_torchscript(
            self, input_spec: InputSpec | None = None, check_trace: bool = True
        ):
            """
            Converts the torch module to a torchscript trace, which
            is the format expected by qai hub.

            This is a default implementation that may be overriden by a subclass.
            """
            if not input_spec:
                input_spec = self.get_input_spec()

            for param in self.parameters():
                param.requires_grad = False

            return torch.jit.trace(
                self, make_torch_inputs(input_spec), check_trace=check_trace
            )

        @staticmethod
        def get_input_spec(
            sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
            context_length: int = DEFAULT_CONTEXT_LENGTH,
        ) -> InputSpec:
            num_layers = NUM_LAYERS_TRUNC if truncate_model else NUM_LAYERS
            return Llama3Base._get_input_spec(
                num_hidden_layers=num_layers,
                sequence_length=sequence_length,
                context_length=context_length,
                hidden_size=HIDDEN_SIZE,
                num_key_value_heads=NUM_KEY_VALUE_HEADS,
                num_attention_heads=NUM_ATTN_HEADS,
            )

    model_cls = Llama3_2_PiQaro_FP
    model_name = MODEL_ID + f"_{args.opt}"

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
        "Samsung Galaxy S23 (Family)",
        "Samsung Galaxy S24 (Family)",
        "Snapdragon X Elite CRD",
    ]

    for i, device in enumerate(devices):
        # Disable for the first device, reuse for all other devices.
        cache = CacheMode.DISABLE if i == 0 else CacheMode.ENABLE
        export_model(
            model_cls=model_cls,
            model_name=model_name,
            model_asset_version=MODEL_ASSET_VERSION,
            components=all_components,
            sub_components=all_sub_components,
            num_layers_per_split=num_layers_per_split,
            device=device,
            output_dir=output_dir,
            _skip_optimizations=skip_optimizations,
            skip_inferencing=True,
            skip_downloading=True,
            model_cache_mode=cache,
            # _skip_quantsim_creation=True,
        )
