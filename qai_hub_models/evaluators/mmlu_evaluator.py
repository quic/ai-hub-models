# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import textwrap
from typing import TYPE_CHECKING, Callable

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.evaluators.base_evaluators import (
    BaseEvaluator,
    MetricMetadata,
    _DataLoader,
)
from qai_hub_models.evaluators.kldiv_evaluator import KLDivEvaluator

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.generator import LLM_Generator


class MMLUEvaluator(BaseEvaluator):
    """Evaluator for computing MMLU of a Large Language Model.
    This may not be as generic as hoped and may need work. Works with Llama 3.2 3B.
    """

    def __init__(
        self,
        context_length: int,
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
    ):
        self.context_length = context_length
        self.device = device
        self.choices = self._get_choices(tokenizer)
        self.tokenizer = tokenizer
        # We surface KL divergence too, out of convenience.
        self.kldiv_evaluator = KLDivEvaluator(
            context_length=context_length,
            device=device,
        )
        self.reset()

    @property
    def is_distance_metric(self) -> bool:
        return True

    @staticmethod
    def _get_choices(
        tokenizer: PreTrainedTokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def tokenize_letter(letter: str):
            return tokenizer(letter, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0, -1:]

        return torch.cat(
            [
                tokenize_letter("Answer: A"),
                tokenize_letter("Answer: B"),
                tokenize_letter("Answer: C"),
                tokenize_letter("Answer: D"),
            ],
            dim=-1,
        )

    def add_batch(
        self,
        output: CausalLMOutputWithPast,
        gt: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ):
        is_distance = isinstance(gt, tuple)
        if is_distance:
            self.kldiv_evaluator.add_batch(output, gt)
            self.is_distance = True
            gt, output_fp = gt

        self.batch_index += 1
        assert output.logits is not None
        logits = output.logits[0]

        top_token_id = logits[-1].argmax()
        answers = logits[:, self.choices]
        index = answers[-1].argmax()
        prediction = self.choices[index]
        self.top_is_valid += int(top_token_id in self.choices)

        correct = prediction == gt
        self.correct_answers += int(correct)

        logsoft_q = F.log_softmax(logits, dim=-1)
        self.neg_log_likelihood += float(-logsoft_q[-1, gt])

        if is_distance:
            rest_choices = torch.tensor(
                [c for c in range(logits.shape[-1]) if c not in self.choices]
            )

            logits_fp = output_fp[0]

            top_token_id_fp = logits_fp[-1].argmax()
            answers_fp = logits_fp[:, self.choices]
            index_fp = answers_fp[-1].argmax()
            prediction_fp = self.choices[index_fp]
            self.top_is_valid_fp += int(top_token_id_fp in self.choices)

            correct_fp = prediction_fp == gt
            self.correct_answers_fp += int(correct_fp)
            self.flips += int(correct != correct_fp)

            logsoft_fp = F.log_softmax(logits_fp, dim=-1)

            # Compute KL divergence, collapsing non-answers to "rest"
            logsoft_fp_answers = logsoft_fp[-1, self.choices]
            logsoft_fp_rest = logsoft_fp[-1, rest_choices]
            logsoft_fp_5 = torch.cat(
                [
                    logsoft_fp_answers,
                    logsoft_fp_rest.logsumexp(dim=-1, keepdims=True),
                ],
                dim=-1,
            )

            logsoft_q_answers = logsoft_q[-1, self.choices]
            logsoft_q_rest = logsoft_q[-1, rest_choices]
            logsoft_q_5 = torch.cat(
                [
                    logsoft_q_answers,
                    logsoft_q_rest.logsumexp(dim=-1, keepdims=True),
                ],
                dim=-1,
            )

            kl_scale = math.log(5)
            kldiv_5 = (
                F.kl_div(
                    logsoft_q_5, logsoft_fp_5, reduction="none", log_target=True
                ).sum(-1)
                / kl_scale
            )
            rev_kldiv_5 = (
                F.kl_div(
                    logsoft_fp_5, logsoft_q_5, reduction="none", log_target=True
                ).sum(-1)
                / kl_scale
            )

            self.last_kldiv_5cat += float(kldiv_5)
            self.last_rev_kldiv_5cat += float(rev_kldiv_5)

            self.neg_log_likelihood_fp += float(-logsoft_fp[-1, gt])

    def reset(self):
        self.kldiv_evaluator.reset()

        # Competency metrics (Q)
        self.correct_answers = 0
        self.neg_log_likelihood = 0.0
        self.top_is_valid = 0

        # Competency metrics (FP)
        self.correct_answers_fp = 0
        self.neg_log_likelihood_fp = 0.0
        self.top_is_valid_fp = 0

        # Distances
        self.last_kldiv_5cat = 0.0
        self.last_rev_kldiv_5cat = 0.0
        self.flips = 0

        self.batch_index = 0
        self.is_distance = False

    def get_accuracy_score(self) -> float:
        return self.correct_answers / self.batch_index

    def get_accuracy_score_fp(self) -> float:
        return self.correct_answers_fp / self.batch_index

    def get_avg_neg_log_likelihood(self) -> float:
        return self.neg_log_likelihood / self.batch_index

    def get_avg_neg_log_likelihood_fp(self) -> float:
        return self.neg_log_likelihood_fp / self.batch_index

    def get_avg_last_kldiv_5cat(self) -> float:
        return self.last_kldiv_5cat / self.batch_index

    def get_avg_last_rev_kldiv_5cat(self) -> float:
        return self.last_rev_kldiv_5cat / self.batch_index

    def get_avg_flips(self) -> float:
        return self.flips / self.batch_index

    def get_avg_valid_answers(self) -> float:
        return self.top_is_valid / self.batch_index

    def get_avg_valid_answers_fp(self) -> float:
        return self.top_is_valid_fp / self.batch_index

    def formatted_accuracy(self) -> str:
        if self.is_distance:
            return textwrap.dedent(
                f"""
                MMLU (Quantized): {self.get_accuracy_score():.2%} (higher is better)
                MMLU (FP): {self.get_accuracy_score_fp():.2%} (higher is better)
                Flips {self.get_avg_flips():.2%}

                Top prediction is valid answer (Quantized): {self.get_avg_valid_answers():.1%}
                Top prediction is valid answer (FP): {self.get_avg_valid_answers_fp():.1%}

                Avg NLL (Quantized): {self.get_avg_neg_log_likelihood():.3f} (lower is better)
                Avg NLL (FP): {self.get_avg_neg_log_likelihood_fp():.3f} (lower is better)

                KL Divergence (all, log-C): {self.kldiv_evaluator.get_avg_kldiv():.3%}
                KL Divergence (final answer, log-C): {self.kldiv_evaluator.get_avg_last_kldiv():.3f}
                KL Divergence [5-categories] (final answer, log-5): {self.get_avg_last_kldiv_5cat():.3f}

                rev KL Divergence (all, log-C): {self.kldiv_evaluator.get_avg_rev_kldiv():.3%}
                rev KL Divergence (final answer, log-C): {self.kldiv_evaluator.get_avg_last_rev_kldiv():.3f}
                rev KL Divergence [5-categories] (final answer, log-5): {self.get_avg_last_rev_kldiv_5cat():.3f}
            """
            ).lstrip()
        else:
            return textwrap.dedent(
                f"""
                MMLU: {self.get_accuracy_score():.2%} (higher is better)
                Top prediction is valid answer: {self.get_avg_valid_answers():.1%}
                Avg NLL: {self.get_avg_neg_log_likelihood():.3f} (lower is better)
            """
            ).lstrip()

    def for_each_batch(
        self,
        generator: LLM_Generator,
        data: _DataLoader,
        num_samples: int | None = None,
        callback: (
            Callable[[list[torch.tensor], CausalLMOutputWithPast, torch.Tensor], None]
            | None
        ) = None,
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

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Massive Multitask Language Understanding",
            unit="MMLU",
            description="A measure of how well the model can answer multiple choice questions.",
        )
