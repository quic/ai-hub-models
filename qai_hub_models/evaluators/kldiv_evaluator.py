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

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.generator import LLM_Generator


class KLDivEvaluator(BaseEvaluator):
    """
    Evaluator for computing KL divergence between two probability
    distributions (with inputs assumed to be in logit space). Currently only
    works for LLM outputs (assumes CausalLMOutputWithPast object), but can be
    generalized if needed.
    """

    def __init__(
        self,
        context_length: int,
        device: torch.device,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        self.context_length = context_length
        self.device = device
        self.tokenizer = tokenizer
        self.reset()

    @property
    def is_distance_metric(self) -> bool:
        return True

    def add_batch(
        self,
        output: CausalLMOutputWithPast,
        gt: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ):
        """
        output: This is the output of an LLM_Generator, which produces a
            CauslLMOutputWithPast instance.
        gt: Ground truth (gt) tensor, or optionally (gt, target) tuple
            gt: float[1, input_length]
                A single tensor assumed to be the "correct answer".
                This is only well-defined for tasks with unambiguously correct
                answers (e.g., MMLU). In the absence of a correct answer, this
                may be filled with input ids as a placeholder (in which case
                a ValueError is raised).

            target: float[1, input_length, vocab_size]
                Output (logits) of a "target model". This is typically an
                unquantized counterpart when a quantized model is evaluated.
                This value is the same size and type as output.logits.
        """
        is_distance = isinstance(gt, tuple)
        if is_distance:
            _, target = gt
        else:
            target = gt

        self.batch_index += 1
        assert output.logits is not None
        logits = output.logits[0]

        q_logsoft = F.log_softmax(logits, dim=-1)

        logits_p = target[0]

        if not logits_p.dtype.is_floating_point:
            # This metric cannot be used on non-float targets
            raise ValueError("KLDivEvaluator requires floating point targets")

        p_logsoft = F.log_softmax(logits_p, dim=-1)

        if self.tokenizer is not None:
            top_token = self.tokenizer.decode(q_logsoft[-1].argmax())
            top_token_fp = self.tokenizer.decode(p_logsoft[-1].argmax())

            self.flips += int(top_token != top_token_fp)

        # KL divergence over entire vocabulary size
        kl_scale = math.log(logits.shape[-1])
        kldiv = (
            F.kl_div(q_logsoft, p_logsoft, reduction="none", log_target=True).sum(-1)
            / kl_scale
        )
        # Also compute reverse KL div (which penalizes over-confident responses)
        rev_kldiv = (
            F.kl_div(p_logsoft, q_logsoft, reduction="none", log_target=True).sum(-1)
            / kl_scale
        )
        # Average all time steps
        self.kldiv += float(kldiv.mean())
        self.rev_kldiv += float(rev_kldiv.mean())

        # In some situations, the final prediction is of particular importance.
        # For instance in datasets like MMLU (where it is the correct answer),
        # or in prompts that have been cut-off at a particularly problematic
        # token.
        self.last_kldiv += float(kldiv[-1])
        self.last_rev_kldiv += float(rev_kldiv[-1])

    def reset(self):
        # Distances
        self.kldiv = 0.0
        self.last_kldiv = 0.0
        self.rev_kldiv = 0.0
        self.last_rev_kldiv = 0.0
        self.flips = 0

        self.batch_index = 0

    def get_accuracy_score(self) -> float:
        return self.get_avg_last_kldiv()

    def get_avg_last_kldiv(self) -> float:
        return self.last_kldiv / self.batch_index

    def get_avg_kldiv(self) -> float:
        return self.kldiv / self.batch_index

    def get_avg_last_rev_kldiv(self) -> float:
        return self.last_rev_kldiv / self.batch_index

    def get_avg_rev_kldiv(self) -> float:
        return self.rev_kldiv / self.batch_index

    def formatted_accuracy(self) -> str:
        if self.batch_index > 0:
            ret = textwrap.dedent(
                f"""
                KL Divergence (all, log-C): {self.get_avg_kldiv():.3%}
                KL Divergence (final answer, log-C): {self.get_avg_last_kldiv():.3%}

                rev KL Divergence (all, log-C): {self.get_avg_rev_kldiv():.3%}
                rev KL Divergence (final answer, log-C): {self.get_avg_last_rev_kldiv():.3%}
            """
            ).lstrip()
            if self.tokenizer is not None:
                ret += f"\nFlips: {self.flips / self.batch_index:.1%}\n"
            return ret
        else:
            return "KL Divergence: Nothing collected."

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
            name="KL divergence",
            unit="kldiv",
            description="A distance metric between two probability distributions.",
        )
