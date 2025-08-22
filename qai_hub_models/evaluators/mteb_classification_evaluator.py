# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from qai_hub_models.datasets.amazon_counterfactual import (
    AmazonCounterfactualClassificationDataset,
    DatasetSplit,
)
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    def __init__(
        self,
        model,
        seq_len: int = 128,
        max_iter: int = 100,
        n_experiments: int = 10,
        samples_per_label: int = 32,
        seed: int = 42,
    ):
        self.max_iter = max_iter
        self.n_experiments = n_experiments
        self.samples_per_label = samples_per_label
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=seq_len
        )

        self.model = model
        train_ds = AmazonCounterfactualClassificationDataset(
            split=DatasetSplit.TRAIN
        ).ds
        self.train_text = train_ds["text"]
        self.y_train = train_ds["label"]
        self.train_model()
        self.reset()

    def reset(self):
        self.test_text = []
        self.y_test = []

    def add_batch(self, out: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            out: torch.Tensor
                Transformer embeddings of shape [1, 512], dtype of fp32

            gt: torch.Tensor
                label with shape [1,], dtype of int [0 or 1]
                0: not-counterfactual
                1: counterfactual
        """
        for i in range(out.shape[0]):
            self.test_text.append(out[i].unsqueeze(0))
            self.y_test.append(gt[i].unsqueeze(0))

    def _undersample_data(
        self,
        X: list[str],
        y: list[int],
        samples_per_label: int,
        idxs: np.ndarray | None = None,
    ) -> tuple[list[str], list[int], np.ndarray]:
        """
        Undersample data to have samples_per_label samples of each label

        Args:
            X: list[str]
                List of samples text
            y: list[int]
                List of labels 0 or 1
            samples_per_label: int
                Number of samples per label to undersample to
            idxs: np.ndarray
                List of indices of the samples to undersample from

        returns:
            X_sampled: list[str]
                List of under_sampled samples text
            y_sampled: list[int]
                List of under_sampled labels 0 or 1
            idxs: np.ndarray
                List of indices of the under_sampled samples
        """
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        label_counter: DefaultDict[str, int] = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs

    def train_model(self):
        idxs = None
        self.trained_model = []
        for _ in range(self.n_experiments):
            train_text, y_train, idxs = self._undersample_data(
                self.train_text,
                self.y_train,
                self.samples_per_label,
                idxs,
            )

            clf = LogisticRegression(
                random_state=self.seed,
                n_jobs=-1,
                max_iter=self.max_iter,
            )
            X_train_list = []
            for st in train_text:
                text = "classification: " + st
                inputs = self.tokenizer(text, padding="max_length", return_tensors="pt")
                input_ids = cast(torch.Tensor, inputs["input_ids"])
                attention_mask = cast(torch.Tensor, inputs["attention_mask"])
                X_train_list.append(self.model(input_ids, attention_mask))
            X_train = np.concatenate(X_train_list)

            clf.fit(X_train, y_train)
            self.trained_model.append(clf)

    def get_accuracy_score(self) -> float:
        scores_list = []
        for clf in self.trained_model:
            X_test = torch.concat(self.test_text)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(self.y_test, y_pred)
            scores_list.append(acc)

        acc = np.mean(scores_list)
        return acc

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score() * 100:.3f}% (Top 1)"
