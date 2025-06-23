# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch[0]["input_ids"], batch[0]["attention_mask"], batch[0]["label"]


class MMLU(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        num_fewshot: int = 5,
        split: DatasetSplit = DatasetSplit.TEST,
        fewshot_split: str = "dev",
        num_samples: int = 0,
        seed: int = 42,
    ):
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        else:
            raise ValueError("MMLU dataset currently only supports `test` split")

        self.num_fewshot = num_fewshot
        self.fewshot_split = fewshot_split

        self.dataset = load_dataset(path="cais/mmlu", name="all", split=self.split_str)

        self.dataset = self.dataset.shuffle(seed)

        self.preprocess_dataset()

    def __len__(self) -> int:
        if self.num_samples != 0:
            return self.num_samples
        return len(self.dataset)

    def load_fewshot(self) -> dict[str, str]:
        if self.num_fewshot == 0:
            return {}

        fewshot_split = load_dataset("cais/mmlu", name="all", split=self.fewshot_split)
        grouped_fewshot_questions: dict[str, list[str]] = {}

        def group_fewshot_questions(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]
            answer = chr(ord("A") + sample["answer"])

            if len(grouped_fewshot_questions.get(subject, [])) >= self.num_fewshot:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            formatted_question = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}"
            grouped_fewshot_questions[subject].append(formatted_question)

        fewshot_split.map(group_fewshot_questions)

        for _, questions in grouped_fewshot_questions.items():
            if len(questions) < self.num_fewshot:
                raise ValueError(
                    f"Not enough samples available in split {fewshot_split} to satisfy {self.num_fewshot} fewshot samples."
                )

        def combine_questions(questions):
            formatted_string = ""
            for question in questions:
                formatted_string += question
                formatted_string += "\n\n"
            return formatted_string

        formatted_fewshot_questions = {
            subject: combine_questions(questions)
            for subject, questions in grouped_fewshot_questions.items()
        }
        return formatted_fewshot_questions

    def preprocess_dataset(self):
        fewshot_subject_headers = self.load_fewshot()

        def tokenize(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]

            formatted_question = list(
                map(
                    lambda question, choices: f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:",
                    question,
                    choices,
                )
            )
            fewshot_formatted_question = (
                list(
                    map(
                        lambda subject, question: str(
                            fewshot_subject_headers[subject] + question
                        ),
                        subject,
                        formatted_question,
                    )
                )
                if self.num_fewshot > 0
                else formatted_question
            )

            tokenized_question = self.tokenizer(
                fewshot_formatted_question,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: [field[-self.context_length :]], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = self.tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), sample["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        # if a cache file storing the current computation from function can be identified, use it instead of recomputing.
        map_kwargs = {"num_proc": None, "load_from_cache_file": True}
        self.dataset = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
            ],
            **(map_kwargs if not isinstance(self.dataset, IterableDataset) else {}),
        )

    def __getitem__(self, idx: int):
        return {
            key: torch.Tensor(value).to(dtype=torch.int)
            for key, value in self.dataset[idx].items()
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1
