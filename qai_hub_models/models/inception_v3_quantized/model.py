# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet import (
    AIMETQuantizableMixin,
)

# isort: on

import torch
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim

from qai_hub_models.models.inception_v3.model import InceptionNetV3
from qai_hub_models.utils.aimet.config_loader import get_default_aimet_config
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.quantization_aimet import (
    constrain_quantized_inputs_to_image_range,
    tie_aimet_observer_groups,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 6
DEFAULT_ENCODINGS = "inception_v3_quantized_encodings.json"


class InceptionNetV3Quantizable(
    AIMETQuantizableMixin,
    InceptionNetV3,
):
    """InceptionNetV3 with post train quantization support.

    Supports only 8 bit weights and activations, and only loads pre-quantized checkpoints.
    Support for quantizing using your own weights & data will come at a later date."""

    def __init__(
        self,
        sim_model: QuantizationSimModel,
    ) -> None:
        # Input is already normalized by sim_model. Disable it in the wrapper model.
        InceptionNetV3.__init__(self, sim_model.model, normalize_input=False)
        AIMETQuantizableMixin.__init__(
            self,
            sim_model,
        )

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> "InceptionNetV3Quantizable":
        """
        Parameters:
          aimet_encodings:
            if "DEFAULT": Loads the model with aimet encodings calibrated on imagenette.
            elif None: Doesn't load any encodings. Used when computing encodings.
            else: Interprets as a filepath and loads the encodings stored there.
        """
        model = InceptionNetV3.from_pretrained()
        input_shape = cls.get_input_spec()["image_tensor"][0]

        model = prepare_model(model)
        equalize_model(model, input_shape)
        sim = QuantizationSimModel(
            model,
            quant_scheme="tf_enhanced",
            default_param_bw=8,
            default_output_bw=8,
            config_file=get_default_aimet_config(),
            dummy_input=torch.rand(input_shape),
        )
        cls._tie_pre_concat_quantizers(sim)
        constrain_quantized_inputs_to_image_range(sim)

        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
                ).fetch()
            load_encodings_to_sim(sim, aimet_encodings)

        sim.model.eval()
        return cls(sim)

    @classmethod
    def _tie_pre_concat_quantizers(cls, sim: QuantizationSimModel):
        """
        This ties together the output quantizers prior to concatenations. This
        prevents unnecessary re-quantization during the concatenation, and even
        avoids fatal TFLite converter errors.
        """

        n = sim.model.net
        groups = [
            [
                n.maxpool2,
                n.Mixed_5b.module_avg_pool2d,
            ],
            [
                n.Mixed_5b.branch1x1.module_relu_5,
                n.Mixed_5b.branch5x5_2.module_relu_7,
                n.Mixed_5b.branch3x3dbl_3.module_relu_10,
                n.Mixed_5b.branch_pool.module_relu_11,
                n.Mixed_5b.module_cat,
                n.Mixed_5c.module_avg_pool2d_1,
            ],
            [
                n.Mixed_5c.branch1x1.module_relu_12,
                n.Mixed_5c.branch5x5_2.module_relu_14,
                n.Mixed_5c.branch3x3dbl_3.module_relu_17,
                n.Mixed_5c.branch_pool.module_relu_18,
                n.Mixed_5c.module_cat_1,
                n.Mixed_5d.module_avg_pool2d_2,
            ],
            [
                n.Mixed_5d.branch1x1.module_relu_19,
                n.Mixed_5d.branch5x5_2.module_relu_21,
                n.Mixed_5d.branch3x3dbl_3.module_relu_24,
                n.Mixed_5d.branch_pool.module_relu_25,
                n.Mixed_5d.module_cat_2,
                # This group has a branch with only a max pool,
                # this requires the two concat groups to merge
                n.Mixed_6a.branch3x3.module_relu_26,
                n.Mixed_6a.branch3x3dbl_3.module_relu_29,
                n.Mixed_6a.module_max_pool2d,
                n.Mixed_6a.module_cat_3,
                n.Mixed_6b.module_avg_pool2d_3,
            ],
            [
                n.Mixed_6b.branch1x1.module_relu_30,
                n.Mixed_6b.branch7x7_3.module_relu_33,
                n.Mixed_6b.branch7x7dbl_5.module_relu_38,
                n.Mixed_6b.branch_pool.module_relu_39,
                n.Mixed_6b.module_cat_4,
                n.Mixed_6c.module_avg_pool2d_4,
            ],
            [
                n.Mixed_6c.branch1x1.module_relu_40,
                n.Mixed_6c.branch7x7_3.module_relu_43,
                n.Mixed_6c.branch7x7dbl_5.module_relu_48,
                n.Mixed_6c.branch_pool.module_relu_49,
                n.Mixed_6c.module_cat_5,
                n.Mixed_6d.module_avg_pool2d_5,
            ],
            [
                n.Mixed_6d.branch1x1.module_relu_50,
                n.Mixed_6d.branch7x7_3.module_relu_53,
                n.Mixed_6d.branch7x7dbl_5.module_relu_58,
                n.Mixed_6d.branch_pool.module_relu_59,
                n.Mixed_6d.module_cat_6,
                n.Mixed_6e.module_avg_pool2d_6,
            ],
            [
                n.Mixed_6e.branch1x1.module_relu_60,
                n.Mixed_6e.branch7x7_3.module_relu_63,
                n.Mixed_6e.branch7x7dbl_5.module_relu_68,
                n.Mixed_6e.branch_pool.module_relu_69,
                n.Mixed_6e.module_cat_7,
                # This group has a branch with only a max pool,
                # this requires the two concat groups to merge
                n.Mixed_7a.branch3x3_2.module_relu_71,
                n.Mixed_7a.branch7x7x3_4.module_relu_75,
                n.Mixed_7a.module_max_pool2d_1,
                n.Mixed_7a.module_cat_8,
                n.Mixed_7b.module_avg_pool2d_7,
            ],
            [
                n.Mixed_7b.branch1x1.module_relu_76,
                n.Mixed_7b.branch3x3_2a.module_relu_78,
                n.Mixed_7b.branch3x3_2b.module_relu_79,
                n.Mixed_7b.branch3x3dbl_3a.module_relu_82,
                n.Mixed_7b.branch3x3dbl_3b.module_relu_83,
                n.Mixed_7b.branch_pool.module_relu_84,
                n.Mixed_7b.module_cat_9,
                n.Mixed_7b.module_cat_10,
                n.Mixed_7b.module_cat_11,
                n.Mixed_7c.module_avg_pool2d_8,
            ],
            [
                n.Mixed_7c.branch1x1.module_relu_85,
                n.Mixed_7c.branch3x3_2a.module_relu_87,
                n.Mixed_7c.branch3x3_2b.module_relu_88,
                n.Mixed_7c.branch3x3dbl_3a.module_relu_91,
                n.Mixed_7c.branch3x3dbl_3b.module_relu_92,
                n.Mixed_7c.branch_pool.module_relu_93,
                n.Mixed_7c.module_cat_12,
                n.Mixed_7c.module_cat_13,
                n.Mixed_7c.module_cat_14,
            ],
        ]
        tie_aimet_observer_groups(groups)
