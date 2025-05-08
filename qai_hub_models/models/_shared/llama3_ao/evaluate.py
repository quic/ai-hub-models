# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import sys

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.mmlu import MMLU
from qai_hub_models.datasets.mmlu import collate_fn as mmlu_collate_fn
from qai_hub_models.datasets.mmmlu import MMMLU
from qai_hub_models.datasets.mmmlu import collate_fn as mmmlu_collate_fn
from qai_hub_models.datasets.mmmlu import mmmlu_split_lookup
from qai_hub_models.datasets.tiny_mmlu import TinyMMLU
from qai_hub_models.datasets.tiny_mmlu import collate_fn as tiny_mmlu_collate_fn
from qai_hub_models.datasets.wikitext import WikiText
from qai_hub_models.datasets.wikitext import collate_fn as wikitext_collate_fn
from qai_hub_models.datasets.wikitext_ja import WikiText_Japanese
from qai_hub_models.datasets.wikitext_ja import collate_fn as wikitext_ja_collate_fn
from qai_hub_models.evaluators.mmlu_evaluator import MMLUEvaluator
from qai_hub_models.models._shared.llama3_ao.model import (
    DEFAULT_SEQUENCE_LENGTH,
    RopeEmbedding,
    determine_mode,
    verify_mode_and_checkpoint_match,
)
from qai_hub_models.utils.args import get_model_cli_parser, get_model_kwargs
from qai_hub_models.utils.base_model import BaseModel


def get_dataset(model: torch.nn.Module, task: str, num_samples_mmlu: int):
    rope_embeddings = RopeEmbedding(
        max_length=model.context_length, config=model.llm_config
    )

    # Load dataset.
    if task == "wikitext-ppl":
        dataset = WikiText(
            model.tokenizer,
            rope_embeddings,
            model=model,
            block_size=model.sequence_length,
            context_length=model.context_length,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=wikitext_collate_fn
        )
    elif task == "wikitext-ja-ppl":
        dataset = WikiText_Japanese(
            model.tokenizer,
            rope_embeddings,
            model=model,
            block_size=model.sequence_length,
            context_length=model.context_length,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=wikitext_ja_collate_fn
        )
    elif task == "tiny-mmlu-english":
        dataset = TinyMMLU(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=tiny_mmlu_collate_fn
        )
    elif task == "mmlu":
        dataset = MMLU(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_fewshot=5,
            num_samples=num_samples_mmlu,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=mmlu_collate_fn
        )
    elif "mmmlu" in task:
        language_code = task.replace("mmmlu-", "")
        try:
            split_str = mmmlu_split_lookup[language_code]
        except KeyError as exc:
            raise KeyError(
                'Unable to determine MMMLU language split. Please specify MMMLU task as "mmmlu-<language code>". '
                "Example: --task mmmlu-ja"
            ) from exc

        dataset = MMMLU(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_fewshot=5,
            split=split_str,
            num_samples=num_samples_mmlu,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=mmmlu_collate_fn
        )
    else:
        raise ValueError("Use --help to see available tasks.")
    return dataloader


def get_mmlu_evaluator(model, device):
    # Instantiate the evaluator
    return MMLUEvaluator(
        model.context_length, model.sequence_length, model.tokenizer, device
    )


def llama3_evaluate(
    quantized_model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
):
    parser = get_model_cli_parser(quantized_model_cls)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fp", "quantsim"],
        help="Run the floating point model or simulated quantization.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="wikitext-ppl",
        choices=[
            "wikitext-ppl",
            "wikitext-ja-ppl",
            "tiny-mmlu-english",
            "mmlu",
        ]
        + ["mmmlu-" + language_code for language_code in mmmlu_split_lookup.keys()],
        help="Tasks for evaluation.",
    )
    parser.add_argument(
        "--num-samples-mmlu",
        type=int,
        default=100,
        help="Num of samples to use for MMLU or Multilingual MMLU.",
    )

    args = parser.parse_args()
    mode = args.mode
    user_specified_checkpoint = "--checkpoint" in sys.argv
    if (not user_specified_checkpoint or args.checkpoint == "DEFAULT") and mode is None:
        raise ValueError("--mode must be specified if --checkpoint is not.")

    if args.checkpoint != "DEFAULT":
        if not mode:
            mode = determine_mode(args.checkpoint)
        else:
            verify_mode_and_checkpoint_match(args.checkpoint, mode)

    server_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if server_device.type == "cpu":
        print()
        print(
            "WARNING: Evaluation of this model (floating point or QuantSim) takes a long time on CPU. Doing it on a CUDA enabled machine will be faster."
        )
    seq_len = args.sequence_length
    seq_len_in_filename = 0
    if os.path.exists(args.checkpoint):
        for file in os.listdir(args.checkpoint):
            if file.endswith(".onnx") and "seqlen" in file:
                file_name = os.path.basename(file)
                seq_len_in_filename = file_name.split("_")[-2].replace("seqlen", "")
                seq_len = max(int(seq_len_in_filename), seq_len)

    print()
    if seq_len > DEFAULT_SEQUENCE_LENGTH:
        print(
            f"Using the longest available pre-computed sequence length ({seq_len}) to maximize evaluation speed."
        )
    else:
        print(
            "Using a longer sequence length will improve evaluation speed. Use argument --sequence-length to pass a longer sequence length."
        )

    model_cls = fp_model_cls if mode == "fp" else quantized_model_cls
    kwargs = get_model_kwargs(model_cls, vars(args))
    kwargs["sequence_length"] = seq_len  # type: ignore[index]
    if args.mode != "fp":
        kwargs["server_device"] = server_device  # type: ignore[index]
    # We parsed arguments for quantized model, default for quantized model would be None for floating point model.
    kwargs["checkpoint"] = None if args.checkpoint == "DEFAULT" and mode == "fp" else args.checkpoint  # type: ignore[index]

    model = model_cls.from_pretrained(**kwargs)

    eval_dataloader = get_dataset(model, args.task, args.num_samples_mmlu)

    evaluator = (
        get_mmlu_evaluator(model, server_device)
        if "mmlu" in args.task
        else model.get_evaluator()
    )
    evaluator.add_from_dataset(
        model=model, data=eval_dataloader, eval_iterations=len(eval_dataloader)
    )
    print(evaluator.formatted_accuracy())
