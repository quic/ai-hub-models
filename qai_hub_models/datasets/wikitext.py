# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from itertools import chain

import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models._shared.llama3_ao.app import (
    RopeEmbedding,
    get_past_keyval_with_shift,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


def collate_fn(
    batch: tuple[tuple[torch.tensor, ...], list[torch.tensor]]
) -> tuple[tuple[torch.tensor, ...], list[torch.tensor]]:
    return tuple(batch[0][0]), batch[0][1]


class WikiText(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        rope_embeddings: RopeEmbedding,
        model: BaseModel,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        device: torch.device = torch.device("cpu"),
        num_samples: int = 0,
    ):
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        # Needed to create position ids cos and sin correctly for inference.
        self.rope_embeddings = rope_embeddings
        self.model = model
        self.num_samples = num_samples

        self.input_specs = self.model.get_input_spec(
            sequence_length=self.block_size,
            context_length=self.context_length,
        )
        # Pass KV cache shape
        self.kv_cache = []
        for k, (shape, _) in self.input_specs.items():
            if k.startswith("past_"):
                self.kv_cache.append(torch.zeros(shape))

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        elif split == DatasetSplit.TRAIN:
            self.split_str = "train"
        else:
            raise ValueError(
                "Wikitext dataset currently only supports `test` and `train` split"
            )

        self.load_raw_dataset()
        self.device = device

    def load_raw_dataset(self):
        self.dataset = load_dataset(
            path="wikitext", name="wikitext-2-raw-v1", split=self.split_str
        )
        if self.split_str == "train":
            self._preprocess_train_dataset()
        else:
            self.tokens = self.tokenizer(
                "\n\n".join(self.dataset["text"]),
                return_tensors="pt",
                add_special_tokens=True,
            )

    def __len__(self) -> int:
        if self.num_samples != 0:
            return self.num_samples
        if self.split_str == "train":
            # 80k samples to be passed for calibration and advanced algorithms like Sequential MSE.
            return 20 * self.context_length // self.block_size
        # TODO: #14726 The data is being truncated here and causing maybe an incorrect PPL computation.
        return len(self.tokens["input_ids"][0]) // self.block_size

    def _group_texts(self, examples):
        """
        Main data processing function that will concatenate all texts from train split of the dataset and generate chunks of block_size.
        """
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        # TODO: #14733 Investigate if we cannot drop some tokens here.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _tokenize_fn(self, examples):
        return self.tokenizer(
            examples["text"],
            return_token_type_ids=False,
            add_special_tokens=True,
        )

    def _preprocess_train_dataset(self):
        map_kwargs = {"num_proc": None, "load_from_cache_file": True}

        tokenized_dataset = self.dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=["text"],
            **(map_kwargs if not isinstance(self.dataset, IterableDataset) else {}),
        )

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess. To speed up this part, we use multiprocessing.
        self.dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            **(
                map_kwargs if not isinstance(tokenized_dataset, IterableDataset) else {}
            ),
        )

    def __getitem__(self, idx: int):
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size
        if self.split_str == "train":
            input_ids = torch.tensor(self.dataset["input_ids"][idx]).view(1, -1)
        else:
            input_ids = self.tokens["input_ids"][0, start_idx:end_idx].view(1, -1)
        labels = input_ids.clone()
        num_blocks = int(self.context_length / self.block_size)
        effective_idx = idx % num_blocks

        # Attention mask
        attn_mask = torch.zeros((1, self.context_length))
        attn_mask[
            :, self.context_length - (effective_idx + 1) * self.block_size :
        ] = 1.0

        # Use rope embeddings to get the position ids
        position_ids_lst = list(
            range(
                effective_idx * self.block_size,
                (effective_idx + 1) * self.block_size,
            )
        )
        position_ids = (
            torch.Tensor(position_ids_lst).type(torch.long).reshape(1, self.block_size)
        )
        position_ids_cos, position_ids_sin = self.rope_embeddings.get_embedding(
            position_ids
        )

        mask_neg = -50.0
        mask = torch.full(
            (self.block_size, self.block_size),
            torch.tensor(mask_neg),
        )
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = torch.cat(
            [
                torch.zeros(
                    self.block_size,
                    self.context_length - self.block_size,
                ),
                mask,
            ],
            dim=-1,
        )
        mask[None, None, :, :].expand(1, 1, self.block_size, self.context_length)
        expanded_mask = attn_mask[:, None, None, :].expand(
            1, 1, self.block_size, self.context_length
        )
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), mask_neg
        )
        cm_attn_mask = inverted_mask + mask
        cm_attn_mask.clamp_min(mask_neg)

        kv_cache_shape = []
        for k, (shape, _) in self.input_specs.items():
            if k.startswith("past_"):
                kv_cache_shape.append(shape)
        if self.split_str == "train":
            self.model.to(self.device)
            inputs = [input_ids, cm_attn_mask, position_ids_cos, position_ids_sin]
            inputs.extend([kv for kv in self.kv_cache])
            inputs = [inp.to(self.device) for inp in inputs]
            output = self.model(*inputs)

            # This kv cache is needed to maintain the context between multiple blocks.
            self.kv_cache = (
                [torch.zeros(shape) for shape in kv_cache_shape]
                if effective_idx + 1 == 0
                else get_past_keyval_with_shift(
                    self.kv_cache,
                    output[1:],
                    length=self.context_length - self.block_size,
                )
            )
            results = (
                input_ids,
                cm_attn_mask,
                position_ids_cos,
                position_ids_sin,
                *self.kv_cache,
            ), labels
        else:
            results = (
                input_ids.to(torch.int32),
                cm_attn_mask,
                position_ids_cos,
                position_ids_sin,
            ), labels
        return results

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1


def load_calibration_data(
    split: DatasetSplit,
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    dataset_cls: type[WikiText] = WikiText,
):
    """
    This loads the dataset for calibration. The floating point torch model is passed here so that the
    kv_cache input can be generated since its the output of the previous model.

    """
    rope_embeddings = RopeEmbedding(
        max_length=model.context_length, config=model.llm_config
    )
    dataset = dataset_cls(
        tokenizer=model.tokenizer,
        rope_embeddings=rope_embeddings,
        block_size=model.sequence_length,
        context_length=model.context_length,
        split=split,
        model=model,
        device=device,
        num_samples=num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    input_spec = model.get_input_spec(
        sequence_length=model.sequence_length,
        context_length=model.context_length,
    )
    assert input_spec is not None
    inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(len(input_spec))]

    for (sample_input, _) in dataloader:
        for i, tensor in enumerate(sample_input):
            inputs[i].append(tensor)

    return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
