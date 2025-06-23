# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, _DataLoader
from qai_hub_models.models._shared.llm.generator import LLM_Generator


class MMLUEvaluator(BaseEvaluator):
    """Evaluator for computing MMLU of a Large Language Model.
    This may not be as generic as hoped and may need work. Works with Llama 3.2 3B.
    """

    def __init__(
        self,
        context_length: int,
        block_size: int,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.context_length = context_length
        self.block_size = block_size
        self.device = device
        self.choices = self._get_choices(tokenizer)
        self.tokenizer = tokenizer
        self.reset()

    @staticmethod
    def _get_choices(
        tokenizer: PreTrainedTokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def tokenize_letter(letter: str):
            return torch.Tensor(
                tokenizer(letter, add_special_tokens=False)["input_ids"]
            ).to(dtype=torch.int)

        return (
            tokenize_letter("A"),
            tokenize_letter("B"),
            tokenize_letter("C"),
            tokenize_letter("D"),
        )

    def add_batch(self, output: CausalLMOutputWithPast, gt: torch.Tensor):
        self.batch_index += 1
        logits = output.logits

        lm_logits = logits.reshape(1, -1, logits.shape[-1])
        last_logit = lm_logits[0, -1, :].flatten()

        scores = tuple(last_logit[choice] for choice in self.choices)
        index = scores.index(max(scores))
        prediction = self.choices[index]

        if prediction == gt:
            self.correct_answers += 1

    def reset(self):
        self.correct_answers = 0
        self.batch_index = 0

    def get_accuracy_score(self) -> float:
        return self.correct_answers / self.batch_index

    def formatted_accuracy(self) -> str:
        return f"MMLU: {self.get_accuracy_score():.2f}"

    def for_each_batch(
        self,
        generator: LLM_Generator,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: Callable[
            [list[torch.tensor], CausalLMOutputWithPast, torch.Tensor], None
        ]
        | None = None,
    ) -> None:
        total_samples = 0
        batch_size = 1
        num_samples = num_samples or len(data)
        with tqdm(
            total=num_samples,
            desc="Number of samples completed",
        ) as pbar:
            for sample in data:
                input_ids, attention_mask, ground_truth = sample  # type:ignore[misc]
                inputs = [input_ids, attention_mask]
                inputs = [inp.to(self.device) for inp in inputs]
                outputs = generator(*inputs)
                if callback:
                    callback(inputs, outputs, ground_truth)
                total_samples += 1
                pbar.update(batch_size)
                if total_samples >= num_samples:
                    break

    def add_from_dataset(
        self,
        generator: LLM_Generator,
        data: _DataLoader,
        eval_iterations: int | None = None,
    ) -> None:
        def _add_batch(
            _: list[torch.Tensor],
            outputs: CausalLMOutputWithPast,
            ground_truth: torch.Tensor,
        ):
            self.add_batch(outputs, ground_truth)

        self.for_each_batch(generator, data, eval_iterations, _add_batch)
