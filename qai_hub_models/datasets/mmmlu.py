# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit

mmmlu_splits: list[str] = [
    "mmmlu",
    "mmmlu_ar",
    "mmmlu_bn",
    "mmmlu_de",
    "mmmlu_es",
    "mmmlu_fr",
    "mmmlu_hi",
    "mmmlu_id",
    "mmmlu_it",
    "mmmlu_ja",
    "mmmlu_ko",
    "mmmlu_pt",
    "mmmlu_sw",
    "mmmlu_yo",
    "mmmlu_zh",
]


class MMMLU(BaseDataset):
    SUBSET_NAME = "default"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        num_fewshot: int = 5,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
        seed: int = 42,
    ):
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer

        if split != DatasetSplit.TEST:
            raise ValueError("This dataset only supports the 'test' split")

        self.num_fewshot = num_fewshot
        self.num_samples = num_samples

        self.dataset = load_dataset(
            path="openai/MMMLU", name=self.SUBSET_NAME, split="test"
        )
        self.dataset = self.dataset.shuffle(seed)

        self.preprocess_dataset()

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ]:
        return batch[0]["input_ids"], batch[0]["attention_mask"], batch[0]["label"]

    def __len__(self) -> int:
        if self.num_samples != 0:
            return self.num_samples
        return len(self.dataset)

    def load_fewshot(self) -> dict[str, list[str]]:
        if self.num_fewshot == 0:
            return {}

        grouped_fewshot_questions: dict[str, list[str]] = {}

        def group_fewshot_questions(sample):
            question = sample["Question"]
            choices = (sample["A"], sample["B"], sample["C"], sample["D"])
            subject = sample["Subject"]
            answer = sample["Answer"]

            # We need one extra question to make sure that we can create an appropriately formatted string even if one
            # of the fewshot questions is encountered.
            if len(grouped_fewshot_questions.get(subject, [])) >= self.num_fewshot + 1:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            formatted_question = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}"
            grouped_fewshot_questions[subject].append(formatted_question)

        self.dataset.map(group_fewshot_questions)

        for _, questions in grouped_fewshot_questions.items():
            if len(questions) < self.num_fewshot + 1:
                raise ValueError(
                    f"Not enough samples available in subset '{self.SUBSET_NAME}' to satisfy {self.num_fewshot + 1} fewshot samples."
                )

        return grouped_fewshot_questions

    def preprocess_dataset(self):
        grouped_fewshot_questions = self.load_fewshot()

        def tokenize(sample):
            question = sample["Question"]
            A = sample["A"]
            B = sample["B"]
            C = sample["C"]
            D = sample["D"]
            subject = sample["Subject"]

            formatted_question = list(
                map(
                    lambda question, A, B, C, D: f"{question.strip()}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
                    question,
                    A,
                    B,
                    C,
                    D,
                )
            )

            def assemble_fewshot_question(formatted_question, subject):
                subject_fewshot_questions = grouped_fewshot_questions[subject]

                formatted_string = ""
                num_fewshot_questions_added = 0
                for fewshot_question in subject_fewshot_questions:
                    if num_fewshot_questions_added >= self.num_fewshot:
                        break
                    if formatted_question in fewshot_question:
                        continue

                    formatted_string += fewshot_question
                    formatted_string += "\n\n"
                    num_fewshot_questions_added += 1

                formatted_string += formatted_question
                return formatted_string

            fewshot_formatted_question = list(
                map(assemble_fewshot_question, formatted_question, subject)
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
                list(
                    map(
                        lambda answer: f"Answer: {answer}",
                        sample["Answer"],
                    )
                ),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            # Grab only the last token
            answer_token_ids = tokenized_answer["input_ids"][:, -1:]
            result.update({"label": answer_token_ids})
            return result

        # if a cache file storing the current computation from function can be identified, use it instead of recomputing.
        map_kwargs = {"num_proc": None, "load_from_cache_file": True}
        self.dataset = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=[
                "Question",
                "A",
                "B",
                "C",
                "D",
                "Answer",
                "Subject",
                "Unnamed: 0",
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

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/openai/MMMLU",
            split_description="test split",
        )

    @classmethod
    def dataset_name(cls) -> str:
        if cls.SUBSET_NAME == "default":
            return "mmmlu"
        else:
            return "mmmlu_" + cls.SUBSET_NAME.split("_")[0].lower()


class MMMLU_AR(MMMLU):
    SUBSET_NAME = "AR_XY"


class MMMLU_BN(MMMLU):
    SUBSET_NAME = "BN_BD"


class MMMLU_DE(MMMLU):
    SUBSET_NAME = "DE_DE"


class MMMLU_ES(MMMLU):
    SUBSET_NAME = "ES_LA"


class MMMLU_FR(MMMLU):
    SUBSET_NAME = "FR_FR"


class MMMLU_HI(MMMLU):
    SUBSET_NAME = "HI_IN"


class MMMLU_ID(MMMLU):
    SUBSET_NAME = "ID_ID"


class MMMLU_IT(MMMLU):
    SUBSET_NAME = "IT_IT"


class MMMLU_JA(MMMLU):
    SUBSET_NAME = "JA_JP"


class MMMLU_KO(MMMLU):
    SUBSET_NAME = "KO_KR"


class MMMLU_PT(MMMLU):
    SUBSET_NAME = "PT_BR"


class MMMLU_SW(MMMLU):
    SUBSET_NAME = "SW_KE"


class MMMLU_YO(MMMLU):
    SUBSET_NAME = "YO_NG"


class MMMLU_ZH(MMMLU):
    SUBSET_NAME = "ZH_CN"
