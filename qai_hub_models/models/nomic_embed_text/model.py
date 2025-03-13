# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, dynamic_module_utils

from qai_hub_models.utils.asset_loaders import PathLike
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_MODEL_VERSION = "1.5"
MATRYOSHIKA_DIM = 512


# Modify transformers so that it creates modules that can be traced.
# See https://github.com/huggingface/transformers/issues/35570 for details.
# TODO: Remove this when transfomers releases the upstreamed patch.
def _patched_transformers_get_class_in_module(
    class_name: str,
    module_path: PathLike,
) -> type:
    name = os.path.normpath(module_path).rstrip(".py").replace(os.path.sep, ".")
    # Everything in this function is copied from transformers except the following ".".join() statement.
    name = ".".join(
        [
            f"_{x}" if x and x[0].isdigit() else x
            for x in name.replace("-", "_").split(os.path.sep)
        ]
    )
    module_spec = importlib.util.spec_from_file_location(
        name, location=Path(dynamic_module_utils.HF_MODULES_CACHE) / module_path
    )
    if not module_spec or not module_spec.loader:
        raise ValueError(f"Module spec not found for path {name}")

    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(module_spec)
        # insert it into sys.modules before any loading begins
        sys.modules[name] = module

    # reload in both cases
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)


class NomicEmbedText(BaseModel):
    def __init__(self, model: nn.Module, model_version: str, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.model = model
        self.model_version = model_version
        assert model_version in [DEFAULT_MODEL_VERSION, "1"]
        model.eval()

    @classmethod
    def from_pretrained(
        cls, model_version: str = DEFAULT_MODEL_VERSION, sequence_length: int = 128
    ) -> NomicEmbedText:
        """
        Create a Nomic Embedding BERT model.

        Parameters:
            model_version: str
                Version of the Nomic model (1 or 1.5)

            sequence_length: int
                Model max sequence length
                (When compiled for device, the sequence must be padded to this size.)
        """
        with mock.patch(
            "transformers.dynamic_module_utils.get_class_in_module",
            _patched_transformers_get_class_in_module,
        ):
            return cls(
                AutoModel.from_pretrained(
                    f"nomic-ai/nomic-embed-text-v{model_version}",
                    trust_remote_code=True,
                    rotary_scaling_factor=sequence_length // 2048 or 1,
                ),
                model_version,
                sequence_length,
            )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Produce embeddings from the given tokenized inputs.

        Parameters:
            input_ids: torch.Tensor
                Tokenized inputs of shape (1, sequence_length), dtype of int32

            attention_mask: torch.Tensor
                Attention mask of shape (1, sequence_length), dtype of fp32

            Where the default value of sequence_length is 128.

        Returns:
            token_embeddings: torch.Tensor
                Transformer embeddings of shape [1, 512], dtype of fp32
        """
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)[0]
        token_embeddings = NomicEmbedText._mean_pooling(
            last_hidden_state, attention_mask
        )
        if self.model_version == "1.5":
            token_embeddings = F.layer_norm(
                token_embeddings, normalized_shape=(token_embeddings.shape[1],)
            )
            token_embeddings = token_embeddings[:, :MATRYOSHIKA_DIM]
        return F.normalize(token_embeddings, p=2, dim=1)

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sequence_length: int = 128,
    ) -> InputSpec:
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "input_tokens": ((batch_size, sequence_length), "int32"),
            "attention_masks": ((batch_size, sequence_length), "float32"),
        }

    def _get_input_spec_for_instance(self, batch_size: int = 1) -> InputSpec:
        return NomicEmbedText.get_input_spec(batch_size, self.seq_length)

    @staticmethod
    def get_output_names() -> list[str]:
        return ["embeddings"]
