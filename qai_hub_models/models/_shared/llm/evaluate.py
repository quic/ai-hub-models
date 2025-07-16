# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import importlib

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets import get_dataset_from_name
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.mmmlu import mmmlu_split_lookup
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.model import is_quantized_checkpoint
from qai_hub_models.utils.args import get_model_cli_parser, get_model_kwargs
from qai_hub_models.utils.base_model import BaseModel


def get_dataset(model: torch.nn.Module, task: str, num_samples: int):
    # Get dataset by name
    eval_dataset = task.replace("-ppl", "").replace("-", "_")
    kwargs = dict(
        tokenizer=model.tokenizer,
        block_size=model.sequence_length,
        context_length=model.context_length,
        num_samples=num_samples,
        split=DatasetSplit.TEST,
    )

    # Load dataset.
    if eval_dataset not in {"wikitext", "wikitext_ja", "tiny_mmlu"}:
        kwargs["num_fewshot"] = 5
    elif eval_dataset.startswith("mmmlu"):
        language_code = task.replace("mmmlu-", "")
        try:
            split_str = mmmlu_split_lookup[language_code]
        except KeyError as exc:
            raise ValueError(
                'Unable to determine MMMLU language split. Please specify MMMLU task as "mmmlu-<language code>". '
                "Example: --task mmmlu-ja"
            ) from exc
        kwargs["split"] = split_str

    dataset = get_dataset_from_name(
        name=eval_dataset,
        **kwargs,
    )
    # Import correct collate_fn
    python_folder = eval_dataset.replace("-", "_")
    module_name = f"qai_hub_models.datasets.{python_folder}"
    dataset_module = importlib.import_module(module_name)
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=1, collate_fn=dataset_module.collate_fn
    )
    return dataloader


def llm_evaluate(
    quantized_model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
    default_calibration_seqlen: int = 2048,
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
    parser.set_defaults(sequence_length=default_calibration_seqlen)
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
    evaluator = model.get_evaluator(
        args.task, torch.device("cpu") if is_quantized else host_device
    )

    embedding = fp_model_cls.EmbeddingClass(
        max_length=args.context_length,
        config=model.llm_config,
    )
    generator = LLM_Generator([model], model.tokenizer, embedding)

    evaluator.add_from_dataset(
        generator=generator, data=eval_dataloader, eval_iterations=len(eval_dataloader)
    )
    print(evaluator.formatted_accuracy())
