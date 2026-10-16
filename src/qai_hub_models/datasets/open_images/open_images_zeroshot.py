# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.datasets.imagenet.imagenet_zeroshot import (
    imagenet_prompts,
    tokenize_prompts,
)
from qai_hub_models.datasets.open_images.open_images import OpenImagesV7Dataset
from qai_hub_models.utils.base_dataset import DatasetSplit


class OpenImagesV7ZeroshotDataset(OpenImagesV7Dataset):
    """Open Images V7 dataset with optional joint image+text mode for calibration.

    Parameters
    ----------
    split
        Dataset split. Only ``TRAIN`` is supported.
    tokenizer_id
        Hugging Face model id whose tokenizer is used to encode class prompts.
        When ``None`` (default) ``__getitem__`` returns ``(image, label)``.
        When provided ``__getitem__`` returns ``(image, input_ids, label)``.
    text_seq_len
        Fixed sequence length for tokenized prompts. Only used when
        ``tokenizer_id`` is not ``None``.

    Notes
    -----
    Open Images has no ImageNet labels, so in joint mode sample ``i`` is paired
    with the prompt of ImageNet-1K class ``i % 1000``. The pairing is arbitrary
    and only suitable for calibration, where just the input distribution matters.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        tokenizer_id: str | None = None,
        text_seq_len: int = 64,
    ) -> None:
        super().__init__(split=split)
        self._class_input_ids: torch.Tensor | None = None
        if tokenizer_id is not None:
            self._class_input_ids = tokenize_prompts(
                imagenet_prompts(), tokenizer_id, text_seq_len
            )

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor, int]:
        """
        Parameters
        ----------
        index
            Sample index into the dataset.

        Returns
        -------
        sample : tuple[torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor, int]
            ``(image, label)`` in image-only mode, or ``(image, input_ids, label)``
            in joint mode, where ``label`` is the pseudo class ``index % 1000``.
        """
        image, _ = super().__getitem__(index)
        if self._class_input_ids is not None:
            label = index % len(self._class_input_ids)
            return image, self._class_input_ids[label], label
        return image, 0
