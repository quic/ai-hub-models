# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, _DataLoader
from qai_hub_models.models._shared.llama3_ao.app import (
    RopeEmbedding,
    get_past_keyval_with_shift,
)
from qai_hub_models.models._shared.llama3_ao.model import (
    prepare_combined_attention_mask,
)


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

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        self.batch_index += 1
        logits = output[0]

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
        model: torch.nn.Module,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: Callable[[list[torch.tensor], torch.Tensor, torch.Tensor], None]
        | None = None,
    ) -> None:
        model.to(self.device)
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
                outputs = self.slice_and_shift_inputs(model, *inputs)
                if callback:
                    callback(inputs, outputs, ground_truth)
                total_samples += 1
                pbar.update(batch_size)
                if total_samples >= num_samples:
                    break

    def add_from_dataset(
        self,
        model: torch.nn.Module,
        data: _DataLoader,
        eval_iterations: int | None = None,
    ) -> None:
        def _add_batch(
            _: list[torch.Tensor], outputs: torch.Tensor, ground_truth: torch.Tensor
        ):
            self.add_batch(outputs, ground_truth)

        self.for_each_batch(model, data, eval_iterations, _add_batch)

    def slice_and_shift_inputs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Helper function to slice the input into chunks, and perform inference on each one
        # while maintaining the KV cache context

        input_specs = model.get_input_spec(
            sequence_length=self.block_size,
            context_length=self.context_length,
        )

        rope_embeddings = RopeEmbedding(
            max_length=model.context_length, config=model.llm_config
        )

        kv_cache_shape = []
        for k, (shape, _) in input_specs.items():
            if k.startswith("past_"):
                kv_cache_shape.append(shape)
        kv_cache = [torch.zeros(shape) for shape in kv_cache_shape]

        num_inferences = self.context_length // self.block_size
        for inf_idx in range(num_inferences):
            start_idx = inf_idx * self.block_size
            end_idx = (inf_idx + 1) * self.block_size
            sliced_input_ids = (
                input_ids[start_idx:end_idx]
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.int32)
            )

            sliced_attention_mask = (
                attention_mask[0:end_idx].unsqueeze(0).to(self.device)
            )
            attention_mask_padding = torch.zeros(
                (1, self.context_length - sliced_attention_mask.shape[-1])
            ).to(self.device)
            padded_attention_mask = torch.cat(
                [attention_mask_padding, sliced_attention_mask], dim=-1
            )
            # skip this inference if all tokens are masked
            if torch.all(padded_attention_mask == 0):
                continue

            # Use rope embeddings to get the position ids
            sliced_position_ids = torch.cumsum(attention_mask, dim=-1)[
                start_idx:end_idx
            ]
            position_ids = sliced_position_ids.type(torch.long).reshape(
                1, self.block_size
            )
            position_ids_cos, position_ids_sin = rope_embeddings.get_embedding(
                position_ids
            )

            cm_attn_mask = prepare_combined_attention_mask(
                padded_attention_mask,
                (1, self.block_size),
                self.context_length - self.block_size,
                -50.0,
                torch.float,
            )

            model.to(self.device)
            inputs = [
                sliced_input_ids,
                cm_attn_mask,
                position_ids_cos,
                position_ids_sin,
            ]
            inputs.extend([kv for kv in kv_cache])
            inputs = [inp.to(self.device) for inp in inputs]

            output = model(*inputs)

            # This kv cache is needed to maintain the context between multiple blocks.
            kv_cache = get_past_keyval_with_shift(
                kv_cache,
                output[1:],
                length=self.context_length - self.block_size,
            )

        return output[0]
