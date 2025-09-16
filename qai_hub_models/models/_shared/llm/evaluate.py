# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import gc
from collections.abc import Mapping
from copy import copy
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache

from qai_hub_models.datasets import get_dataset_from_name
from qai_hub_models.datasets.common import AugmentedLabelDataset, DatasetSplit
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.model import is_quantized_checkpoint
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.args import get_model_cli_parser, get_model_kwargs
from qai_hub_models.utils.base_model import BaseModel


def get_dataset(model: torch.nn.Module, task: str, num_samples: int):

    # Get dataset by name
    kwargs = dict(
        tokenizer=model.tokenizer,
        block_size=model.sequence_length,
        context_length=model.context_length,
        num_samples=num_samples,
        split=DatasetSplit.TEST,
    )

    # Load dataset.
    dataset = get_dataset_from_name(
        name=task,
        **kwargs,
    )
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=1, collate_fn=dataset.collate_fn
    )
    return dataloader


def evaluate(
    quantized_model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
    num_samples: int,
    task: str,
    kwargs: Mapping[str, Any],
) -> tuple[float, str]:

    is_quantized = is_quantized_checkpoint(kwargs["checkpoint"])

    host_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if host_device.type == "cpu":
        print()
        print(
            "WARNING: Evaluation of this model (floating point or QuantSim) takes a long time on CPU. Doing it on a CUDA-enabled machine will be faster."
        )

    is_default = str(kwargs["checkpoint"]).startswith("DEFAULT")

    fp_model = fp_model_cls.from_pretrained(  # type: ignore[index]
        sequence_length=kwargs["sequence_length"],
        context_length=kwargs["context_length"],
    ).to(torch.device("cpu"))

    final_kwargs: dict[str, Any] = copy(kwargs)  # type: ignore[arg-type]
    if is_quantized:
        if is_default:
            final_kwargs["fp_model"] = fp_model
        final_kwargs["host_device"] = host_device
        model_cls = quantized_model_cls
    else:
        if "_skip_quantsim_creation" in final_kwargs:
            del final_kwargs["_skip_quantsim_creation"]
        model_cls = fp_model_cls

    if final_kwargs["checkpoint"] in {"DEFAULT", "DEFAULT_UNQUANTIZED"}:
        del final_kwargs["checkpoint"]

    eval_dataloader = get_dataset(fp_model, task, num_samples)
    evaluator = fp_model.get_evaluator(
        task, torch.device("cpu") if is_quantized else host_device
    )

    embedding = None
    if evaluator.is_distance_metric and is_quantized:
        # If it's a distance metric, we run the FP model and attach the outputs
        # to the ground truth of the eval data loader.
        embedding = fp_model_cls.EmbeddingClass(
            max_length=final_kwargs["context_length"],
            config=fp_model.llm_config,
        )

        fp_generator = LLM_Generator(
            [fp_model.to(host_device)],
            fp_model.tokenizer,
            embedding,
            accumulate_logits_on_cpu=True,
        )

        fp_logits_list = []
        for input_ids, attention_mask, *_ in eval_dataloader:
            input_ids = input_ids.to(host_device)
            attention_mask = attention_mask.to(host_device)
            fp_logits = fp_generator(input_ids, attention_mask, DynamicCache()).logits
            fp_logits_list.append(fp_logits.detach().cpu())

        # Augment dataloader
        dataset = AugmentedLabelDataset(eval_dataloader.dataset, fp_logits_list)
        eval_dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=eval_dataloader.batch_size,
            collate_fn=eval_dataloader.collate_fn,
        )

    if is_quantized:
        fp_model.to(torch.device("cpu"))
        if "fp_model" not in final_kwargs:
            del fp_model
            gc.collect()
            torch.cuda.empty_cache()

        model = model_cls.from_pretrained(**final_kwargs).to(host_device)
    else:
        model = fp_model.to(host_device)

    if eval_dataloader is None:
        eval_dataloader = get_dataset(model, task, num_samples)
    if embedding is None:
        embedding = fp_model_cls.EmbeddingClass(
            max_length=final_kwargs["context_length"],
            config=model.llm_config,
        )

    generator = LLM_Generator(
        [model],
        model.tokenizer,
        embedding,
        accumulate_logits_on_cpu=True,
    )

    evaluator.add_from_dataset(
        generator=generator,
        data=eval_dataloader,
        eval_iterations=len(eval_dataloader),
    )
    return evaluator.get_accuracy_score(), evaluator.formatted_accuracy()


def llm_evaluate(
    quantized_model_cls: type[BaseModel],
    fp_model_cls: type[BaseModel],
    supported_precisions: list[Precision],
    default_calibration_seqlen: int = 2048,
):
    parser = get_model_cli_parser(
        quantized_model_cls,
        suppress_help_arguments=["--host-device", "--fp-model", "--precision"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="wikitext",
        choices=fp_model_cls.eval_datasets(),
        help="Tasks for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to be used for evaluation.",
    )

    # Use a higher default sequence length for efficiency
    parser.set_defaults(sequence_length=default_calibration_seqlen)
    args = parser.parse_args()

    kwargs = get_model_kwargs(quantized_model_cls, vars(args))

    _, formatted_accuracy = evaluate(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        num_samples=args.num_samples,
        task=args.task,
        kwargs=kwargs,
    )

    print(formatted_accuracy)
