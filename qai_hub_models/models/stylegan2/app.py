# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.stylegan2.model import StyleGAN2


class StyleGAN2App:
    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
        | Callable[[torch.Tensor], torch.Tensor],
        output_dims: int = 512,
        num_classes: int = 0,
    ):
        self.model = model
        self.output_dims = output_dims
        self.num_classes = num_classes

    def generate_random_vec(self, batch_size=1, seed=None) -> torch.Tensor:
        if isinstance(self.model, StyleGAN2):
            input_spec = self.model.get_input_spec(batch_size)
            return torch.from_numpy(
                self.model.sample_inputs(input_spec, seed=seed)["image_noise"][0]
            )
        return torch.from_numpy(
            np.random.RandomState(seed)
            .randn(batch_size, self.output_dims)
            .astype(np.float32)
        )

    def predict(self, *args, **kwargs):
        # See generate_images.
        return self.generate_images(*args, **kwargs)

    def generate_images(
        self,
        image_noise: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        raw_output: bool = False,
    ) -> torch.Tensor | List[Image.Image]:
        """
        Generate an image.

        Inputs:
            image_noise: torch.Tensor | None
                Random state vector from which images should be generated.
                Shape: [N, self.output_dims]

            class_idx: int | torch.tensor | None
                Class index[es] to generate. If the model was not trained on more than 1
                class, this is unused.

                If an integer, generate all batches with the class index defined by the integer.

                If a tensor, provide tensor of either shape:
                [N, self.num_classes].
                    If a value of class_idx[b, n] is 1, that class will be generated.
                    A maximum of 1 class can be set to 1 per batch.
                [N]
                    Each element is a class index.
                    Generate one batch for each provided class index.

            raw_output:
                If true, returns a tensor of N generated RGB images. It has shape [N, 3, self.output_dims, self.output_dims].
                Otherwise, returns List[PIL.Image]

        Returns:
            See raw_output parameter description.
        """
        if image_noise is None:
            image_noise = self.generate_random_vec(
                batch_size=class_idx.shape[0] if class_idx is not None else 1
            )

        if self.num_classes != 0:
            if isinstance(class_idx, int):
                class_idx = torch.Tensor([class_idx] * image_noise.shape[0])

            if isinstance(class_idx, torch.Tensor) and len(class_idx.shape) == 1:
                # Convert from [N] class index to one-hot [N, # of classes]
                assert class_idx.dtype == torch.int
                model_classes = torch.nn.functional.one_hot(class_idx, self.num_classes)
            else:
                model_classes = class_idx

            image_tensor = self.model(image_noise, model_classes)
        else:
            image_tensor = self.model(image_noise)

        image_tensor = (
            (image_tensor.permute(0, 2, 3, 1) * 127.5 + 128)
            .clamp(0, 255)
            .to(torch.uint8)
        )

        if raw_output:
            return image_tensor

        image_list = []
        for image_tensor in image_tensor:
            image_list.append(Image.fromarray(image_tensor.numpy(), "RGB"))
        return image_list
