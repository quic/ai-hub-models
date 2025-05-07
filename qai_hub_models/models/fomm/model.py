# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Optional

import torch

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_path,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import InputSpec

FOMM_SOURCE_REPOSITORY = "https://github.com/AliaksandrSiarohin/first-order-model/"
FOMM_SOURCE_REPO_COMMIT = "f4ff6da1ef5c0e6bcf6ec80324fab37c92193e84"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = CachedWebModelAsset(
    "https://github.com/AliaksandrSiarohin/first-order-model/raw/master/config/vox-256.yaml",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-256.yaml",
)
DEFAULT_WEIGHTS_GDRIVE = CachedWebModelAsset.from_google_drive(
    "1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-cpk.pth.tar",
)
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "vox-cpk.pth.tar",
)


class FOMMDetector(BaseModel):
    """Keypoint detector that finds keypoints over source and driving images"""

    def __init__(self, kp_detector: torch.nn.Module):
        super().__init__()
        self.model = kp_detector

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run FOMM keypoint detector on an image

        Parameters:
            image: torch.Tensor
                   BxCxHxW
                   Image to detect keypoints in

        Returns:
            keypoints: torch.Tensor
                       B x Num keypoints x 2
                       Keypoints detected in the image
            jacobian:  torch.Tensor
                       B x Num keypoints x 2 x 2
                       Jacobian matrix around each keypoint
        """
        result = self.model(image)
        keypoints = result["value"]
        jacobian = result["jacobian"]
        return keypoints, jacobian

    @classmethod
    def from_pretrained(cls) -> FOMMDetector:
        return FOMM.from_pretrained().detector

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["keypoints", "jacobian"]


class FOMMGenerator(BaseModel):
    """Given keypoints from a source image, a target image, and the norm of the keypoints from the target,
    generates the new target image"""

    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.model = generator

    def forward(
        self,
        image: torch.Tensor,
        source_keypoint_values: torch.Tensor,
        source_keypoint_jacobians: torch.Tensor,
        kp_norm_values: torch.Tensor,
        kp_norm_jacobians: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the new target image based on the source keypoints and target keypoints

        Parameters:
            image:            torch.Tensor
                              BxCxHxW
                              The source image
            source_keypoint_values: torch.Tensor
                                    B x num keypoints x 2
                                    Keypoints detected in source image
            source_keypoint_jacobians: torch.Tensor
                                       B x num keypoints x 2 x 2
                                       Jacobians around source keypoints
            kp_norm_values:         torch.Tensor
                                    B x num keypoints x 2
                                    Normalised keypoints detected in driving image
            kp_norm_jacobians:         torch.Tensor
                                       B x num keypoints x 2 x 2
                                       Jacobians around driving keypoints


        Returns:
            prediction: torch.Tensor
                        BxCxHxW
                        Predicted output image for the given driving frame keypoints
        """

        # run generator. The underlying model takes in dictionaries
        source_kp = dict(
            value=source_keypoint_values, jacobian=source_keypoint_jacobians
        )
        kp_norm = dict(value=kp_norm_values, jacobian=kp_norm_jacobians)
        out = self.model(image, kp_source=source_kp, kp_driving=kp_norm)
        prediction = out["prediction"]
        # For the purposes of tracing we return only the prediction element of the dictionary
        # as this is the only part that the app uses
        return prediction

    @classmethod
    def from_pretrained(cls) -> FOMMGenerator:
        return FOMM.from_pretrained().generator

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
            "source_keypoint_values": ((batch_size, 10, 2), "float32"),
            "source_keypoint_jacobians": ((batch_size, 10, 2, 2), "float32"),
            "kp_norm_values": ((batch_size, 10, 2), "float32"),
            "kp_norm_jacobians": ((batch_size, 10, 2, 2), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_image"]


@CollectionModel.add_component(FOMMDetector)
@CollectionModel.add_component(FOMMGenerator)
class FOMM(PretrainedCollectionModel):
    """Exportable FOMM for Image Editing"""

    def __init__(self, detector: FOMMDetector, generator: FOMMGenerator):
        super().__init__(detector, generator)
        self.detector = detector
        self.generator = generator

    @classmethod
    def from_pretrained(cls, weights_url: Optional[str] = None):
        with SourceAsRoot(
            FOMM_SOURCE_REPOSITORY,
            FOMM_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):

            # Change filename to avoid import clash with current file
            if os.path.exists("demo.py"):
                os.rename("demo.py", "fomm_demo.py")
            from fomm_demo import load_checkpoints

            # Download default config
            fomm_config = DEFAULT_CONFIG.fetch()

            # Download weights
            with qaihm_temp_dir() as tmpdir:
                weights_path = load_path(weights_url or DEFAULT_WEIGHTS, tmpdir)

                generator, kp_detector = load_checkpoints(
                    config_path=fomm_config, checkpoint_path=weights_path, cpu=True
                )

            generator_model = FOMMGenerator(generator)
            kp_detector_model = FOMMDetector(kp_detector)

            return cls(kp_detector_model, generator_model)
