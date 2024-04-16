# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
    constrain_quantized_inputs_to_image_range,
)

# isort: on

from typing import Optional

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.yolov7.model import DEFAULT_WEIGHTS, YoloV7
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import tie_observers

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_ENCODINGS = "yolov7_quantized_encodings.json"


class YoloV7Quantizable(AIMETQuantizableMixin, YoloV7):
    """Exportable Quantized YoloV7 bounding box detector, end-to-end."""

    def __init__(
        self,
        sim_model: QuantizationSimModel,
    ) -> None:
        # Sim model will already include postprocessing
        torch.nn.Module.__init__(self)
        AIMETQuantizableMixin.__init__(self, sim_model)
        self.model = sim_model.model

    @classmethod
    def from_pretrained(
        cls,
        weights_name: Optional[str] = DEFAULT_WEIGHTS,
        aimet_encodings: str | None = "DEFAULT",
        include_postprocessing: bool = True,
    ) -> "YoloV7Quantizable":
        """Load YoloV7 from a weightfile created by the source YoloV7 repository."""
        fp16_model = YoloV7.from_pretrained(
            weights_name,
            include_postprocessing=include_postprocessing,
            split_output=True,
        )

        input_shape = cls.get_input_spec()["image"][0]

        model = prepare_model(fp16_model)
        equalize_model(model, input_shape)

        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        tie_observers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()
        final_model = cls(sim)
        return final_model

    def forward(self, image: torch.Tensor):
        """
        Run YoloV7Quantizable on `image`, and produce a
            predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (center_x, center_y, w, h)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.


            else if self.split_output:
                output_xy: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [x_center, y_center] (box_coordinates)

                output_wh: torch.Tensor
                    Shape is [batch, num_preds, 2]
                        where, 2 is [width, height] (box_size)

                output_scores: torch.Tensor
                    Shape is [batch, num_preds, j]
                        where j is [confidence (1 element) , # of classes elements]


            else:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, k]
                        where, k = # of classes + 5
                        k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                        and box_coordinates are [x_center, y_center, w, h]
        """
        return self.model(image)
