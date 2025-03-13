# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from transformers import AutoTokenizer

from qai_hub_models.utils.base_model import ExecutableModelProtocol


class NomicEmbedTextApp:

    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Posenet.

    The app uses 1 model:
        * Nomic Embedding Text

    For a given image input, the app will:
        * tokenize the text
        * Run inference
        * Returns transformer embeddings
    """

    def __init__(
        self,
        model: ExecutableModelProtocol[torch.Tensor],
        seq_len: int,
    ):
        self.model = model
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=self.seq_len
        )

    def predict(self, *args, **kwargs):
        # See predict_pose_keypoints.
        return self.predict_embeddings(*args, **kwargs)

    def predict_embeddings(self, text: str) -> torch.Tensor:
        """
        Predicts up to 17 pose keypoints for up to 10 people in the image.

        Parameters:
            text: str
                Text from which embeddings should be generated.

        Returns:
            token_embeddings: torch.Tensor
                The generated transformer embeddings of shape [1, 512], dtype of fp32
        """
        inputs = self.tokenizer(text, padding="max_length", return_tensors="pt")
        input_ids = cast(torch.Tensor, inputs["input_ids"])
        attention_mask = cast(torch.Tensor, inputs["attention_mask"])
        return self.model(input_ids, attention_mask)
