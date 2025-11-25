# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import cast

import torch
from PIL import Image

from qai_hub_models.datasets.mpii import MPIIDataset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class HumanPosesDataset(MPIIDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor
            Preprocessed image tensor ready for model input
        label
            Always 0 since this dataset is for calibration only
        """
        image_dict = self.images_dict_list[index]
        image = Image.open(cast(str, image_dict["image_path"]))
        image = image.resize((self.input_width, self.input_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)

        return image_tensor, 0
