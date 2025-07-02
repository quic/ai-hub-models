# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

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
from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator
from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    RopeEmbedding,
    is_quantized_checkpoint,
)
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.utils.args import get_model_cli_parser, get_model_kwargs
from qai_hub_models.utils.base_model import BaseModel


def get_dataset(model: torch.nn.Module, task: str, num_samples: int):
    # Load dataset.
    if task == "wikitext-ppl":
        dataset = WikiText(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=wikitext_collate_fn
        )
    elif task == "wikitext-ja-ppl":
        dataset = WikiText_Japanese(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=wikitext_ja_collate_fn
        )
    elif task == "tiny-mmlu":
        dataset = TinyMMLU(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_samples=num_samples,
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
            num_samples=num_samples,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=mmlu_collate_fn
        )
    elif "mmmlu" in task:
        language_code = task.replace("mmmlu-", "")
        try:
            split_str = mmmlu_split_lookup[language_code]
        except KeyError as exc:
            raise ValueError(
                'Unable to determine MMMLU language split. Please specify MMMLU task as "mmmlu-<language code>". '
                "Example: --task mmmlu-ja"
            ) from exc

        dataset = MMMLU(
            tokenizer=model.tokenizer,
            block_size=model.sequence_length,
            context_length=model.context_length,
            num_fewshot=5,
            split=split_str,
            num_samples=num_samples,
        )
        dataloader = DataLoader(
            dataset, shuffle=False, batch_size=1, collate_fn=mmmlu_collate_fn
        )
    else:
        raise ValueError(
            f"Task not available: {task}. Use --help to see available tasks."
        )
    return dataloader


def get_mmlu_evaluator(model, device):
    # Instantiate the evaluator
    return MMLUEvaluator(
        model.context_length, model.sequence_length, model.tokenizer, device
    )


def get_ppl_evaluator(model, device):
    return PerplexityEvaluator(
        model.context_length, model.sequence_length, model.tokenizer, device
    )


def llama3_evaluate(
    quantized_model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
):
    parser = get_model_cli_parser(
        quantized_model_cls, suppress_help_arguments=["--host-device", "--fp-model"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="wikitext-ppl",
        choices=[
            "wikitext-ppl",
            "wikitext-ja-ppl",
            "tiny-mmlu",
            "mmlu",
        ]
        + ["mmmlu-" + language_code for language_code in mmmlu_split_lookup.keys()],
        help="Tasks for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to be used for evaluation.",
    )

    # Use a higher default sequence length for efficiency
    parser.set_defaults(sequence_length=DEFAULT_CALIBRATION_SEQ_LEN)
    args = parser.parse_args()
    num_samples = args.num_samples
    task = args.task

    if num_samples is None:
        num_samples = 0 if "ppl" in task else 100

    is_quantized = is_quantized_checkpoint(args.checkpoint)

    host_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if host_device.type == "cpu":
        print()
        print(
            "WARNING: Evaluation of this model (floating point or QuantSim) takes a long time on CPU. Doing it on a CUDA-enabled machine will be faster."
        )

    kwargs = get_model_kwargs(quantized_model_cls, vars(args))
    if is_quantized:
        if args.checkpoint.startswith("DEFAULT"):
            kwargs["fp_model"] = fp_model_cls.from_pretrained(  # type: ignore[index]
                sequence_length=args.sequence_length,
                context_length=args.context_length,
            )
        model_cls = quantized_model_cls
    else:
        del kwargs["_skip_quantsim_creation"]  # type: ignore[index, attr-defined]
        del kwargs["fp_model"]  # type: ignore[index, attr-defined]
        model_cls = fp_model_cls

    kwargs["sequence_length"] = args.sequence_length  # type: ignore[index]

    if kwargs["checkpoint"].startswith("DEFAULT"):
        del kwargs["checkpoint"]  # type: ignore[attr-defined]

    model = model_cls.from_pretrained(**kwargs).to(host_device)

    eval_dataloader = get_dataset(model, task, num_samples)

    evaluator = (
        get_mmlu_evaluator(model, torch.device("cpu") if is_quantized else host_device)
        if "mmlu" in args.task
        else get_ppl_evaluator(
            model, torch.device("cpu") if is_quantized else host_device
        )
    )

    rope_embeddings = RopeEmbedding(
        max_length=model.context_length, config=model.llm_config
    )
    generator = LLM_Generator([model], model.tokenizer, rope_embeddings)

    evaluator.add_from_dataset(
        generator=generator, data=eval_dataloader, eval_iterations=len(eval_dataloader)
    )
    print(evaluator.formatted_accuracy())
