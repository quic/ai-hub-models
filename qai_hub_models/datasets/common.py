# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class to be extended by Datasets used in this repo for quantizing models.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.download_data()

    @final
    def download_data(self) -> None:
        if self._validate_data():
            return
        if self.dataset_path.exists():
            # Data is corrupted, delete and re-download
            if self.dataset_path.is_dir():
                shutil.rmtree(self.dataset_path)
            else:
                os.remove(self.dataset_path)

        print("Downloading data")
        self._download_data()
        print("Done downloading")
        if not self._validate_data():
            raise ValueError("Something went wrong during download.")

    @abstractmethod
    def _download_data(self) -> None:
        """
        Method to download necessary data to disk. To be implemented by subclass.
        """
        pass

    def _validate_data(self) -> bool:
        """
        Validates data downloaded on disk. By default just checks that folder exists.
        """
        return self.dataset_path.exists()

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return cls.__module__.split(".")[-1]
