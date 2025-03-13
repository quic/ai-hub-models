# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.datasets.kinetics400 import Kinetics400Dataset
from qai_hub_models.models._shared.video_classifier.utils import preprocess_video_224


class Kinetics400_224Dataset(Kinetics400Dataset):
    def preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return preprocess_video_224(tensor)
