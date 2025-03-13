# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models._shared.llama.model import RopeEmbedding


def collate_fn(
    batch: list[dict[str, torch.tensor]]
) -> tuple[tuple[torch.tensor, ...], list[torch.tensor]]:
    try:
        # kv_cache management must be done where the model inference is called.
        return (
            batch[0]["input_ids"],
            batch[0]["attention_mask"],
            batch[0]["position_ids_cos"],
            batch[0]["position_ids_sin"],
        ), batch[0]["labels"]
    except Exception:
        return ([], [], [], []), []


class WikiText(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        rope_embeddings: RopeEmbedding,
        block_size: int = 2048,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
    ):
        self.block_size = block_size
        self.context_length = context_length

        # Needed to create position ids cos and sin correctly for inference.
        self.rope_embeddings = rope_embeddings

        if split != DatasetSplit.TEST:
            raise ValueError("Wikitext dataset currently only supports `test` split")
        self.split_str = "test"

        self.dataset = load_dataset(
            path="wikitext", name="wikitext-2-raw-v1", split=self.split_str
        )

        self.tokens = tokenizer(
            "\n\n".join(self.dataset["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

    def __len__(self) -> int:
        return len(self.tokens["input_ids"][0]) // self.block_size

    def __getitem__(self, idx: int) -> dict[str, list[torch.tensor]]:
        device = "cpu"
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size

        input_ids = self.tokens["input_ids"][0, start_idx:end_idx]
        labels = input_ids.clone()
        effective_idx = idx % int(self.context_length / self.block_size)

        # Use rope embeddings to get the position ids
        position_ids_lst = list(
            range(
                effective_idx * self.block_size, (effective_idx + 1) * self.block_size
            )
        )
        position_ids = (
            torch.Tensor(position_ids_lst).type(torch.long).reshape(1, self.block_size)
        )
        position_ids_cos, position_ids_sin = self.rope_embeddings.get_embedding(
            position_ids
        )

        # Attention mask
        attn_mask = torch.zeros((1, self.context_length))
        attn_mask[
            :, self.context_length - (effective_idx + 1) * self.block_size :
        ] = 1.0
        mask_neg = -50
        mask = torch.full(
            (self.block_size, self.block_size),
            torch.tensor(mask_neg, device=device),
            device=device,
        )
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = torch.cat(
            [
                torch.zeros(
                    self.block_size,
                    self.context_length - self.block_size,
                    device=device,
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

        return {
            "input_ids": input_ids.view(1, -1),
            "attention_mask": cm_attn_mask,
            "position_ids_cos": position_ids_cos,
            "position_ids_sin": position_ids_sin,
            "labels": labels.view(1, -1),
        }

    def _download_data(self) -> None:
        pass
