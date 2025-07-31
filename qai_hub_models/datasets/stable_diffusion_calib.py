# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models._shared.stable_diffusion.model import StableDiffusionBase
from qai_hub_models.models._shared.stable_diffusion.utils import (
    load_calib_dataset_entries,
    load_calib_tokens,
    load_unet_calib_dataset_entries,
    make_calib_data,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, CachedWebModelAsset
from qai_hub_models.utils.checkpoint import CheckpointSpec, CheckpointType

PROMPT_ASSET_VERSION = 1

PROMPT_PATH_SD = CachedWebModelAsset.from_asset_store(
    "stable_diffusion_v1_5", PROMPT_ASSET_VERSION, "calibration_prompts_500.txt"
)


class StableDiffusionCalibDatasetBase(BaseDataset, ABC):
    """
    Use stable diffusion models to generate intermediate latents for
    calibrating Unet and VaeDecoder.
    """

    def __init__(
        self,
        sd_cls: type[StableDiffusionBase],
        num_samples: int = 100,
        num_steps: int = 20,
        checkpoint: CheckpointSpec = "DEFAULT",
        split: DatasetSplit = DatasetSplit.VAL,
        host_device: torch.device | str = torch.device("cpu"),
        use_controlnet: bool = False,
        prompt_path: str | os.PathLike = "",
        image_cond_path: str | os.PathLike = "",
    ):
        """
        Args:

        - num_samples: Typically num samples are determined at load time.  But
        here we have to generate the data first. It's possible to generate
        on-demand and cache them but for now keep it simple

        - checkpoint: Specify to use custom weight. By default use the default
        fp weights.

        - split: ignored. Required by get_dataset_from_name
        """
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.checkpoint = checkpoint
        if isinstance(checkpoint, CheckpointType):
            checkpoint_str = checkpoint.name
        else:  # checkpoint is str
            assert isinstance(checkpoint, str)
            checkpoint_str = checkpoint.replace("/", "_")  # Handle / in HF repo

        self.sd_cls = sd_cls
        self.host_device = host_device
        self.use_controlnet = use_controlnet

        self.prompt_path = str(prompt_path or PROMPT_PATH_SD.fetch())
        self.image_cond_path = image_cond_path

        # Version dataset by
        #   - model class (e.g., StableDiffusionV2_1)
        #   - model version (DEFAULT or custom weights)
        #   - num_samples
        #   - num_steps
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            sd_cls.__name__, checkpoint_str, f"data_n{num_samples}_t{num_steps}"
        )
        self.unet_calib_path = self.data_path / "unet_calib.pt"
        self.vae_calib_path = self.data_path / "vae_calib.pt"
        self.controlnet_calib_path = self.data_path / "controlnet_calib.pt"
        BaseDataset.__init__(self, self.data_path, split=DatasetSplit.VAL)

    def __getitem__(self, index):
        # [0] to squash batch dim
        tensors = tuple(torch.tensor(v[index][0]) for v in self.ds.values())
        label = 1  # fake label
        if len(tensors) == 1:
            return tensors[0], label
        return tensors, label

    def __len__(self):
        return len(list(self.ds.values())[0])

    @abstractmethod
    def _validate_data(self) -> bool:
        # Use subclass
        pass

    def _download_data(self) -> None:
        # Generate data by running the torch models
        tokenizer = self.sd_cls.make_tokenizer()
        scheduler = self.sd_cls.make_scheduler(self.checkpoint)

        text_encoder_cls = self.sd_cls.component_classes[0]
        text_encoder_hf = text_encoder_cls.torch_from_pretrained(  # type: ignore
            checkpoint=self.checkpoint, host_device=self.host_device
        )
        unet_cls = self.sd_cls.component_classes[1]
        unet_hf = unet_cls.torch_from_pretrained(  # type: ignore
            checkpoint=self.checkpoint,
            adapt_torch_model_options={"on_device_opt": False},
            host_device=self.host_device,
        )

        controlnet_hf = None
        if self.use_controlnet:
            controlnet_cls = self.sd_cls.component_classes[3]
            controlnet_hf = controlnet_cls.torch_from_pretrained(  # type: ignore
                checkpoint=self.checkpoint,
                adapt_torch_model_options={"on_device_opt": False},
                host_device=self.host_device,
            )

        os.makedirs(self.data_path, exist_ok=True)
        make_calib_data(
            self.unet_calib_path,
            self.vae_calib_path,
            self.prompt_path,
            tokenizer,
            text_encoder_hf,
            unet_hf,
            scheduler,
            num_steps=self.num_steps,
            num_samples=self.num_samples,
            guidance_scale=self.sd_cls.guidance_scale,
            controlnet_hf=controlnet_hf,
            export_path_controlnet=self.controlnet_calib_path,
            image_cond_path=self.image_cond_path,
        )


# StableDiffusionCalibDatasetUnet, StableDiffusionCalibDatasetVae, and
# StableDiffusionCalibDatasetControlNet share the
# same data generation (StableDiffusionCalibDataset._download_data) and
# on-disk caches, but they are loaded differently, resulting in different
# dataset
class StableDiffusionCalibDatasetUnet(StableDiffusionCalibDatasetBase):
    def _validate_data(self) -> bool:
        if not self.unet_calib_path.exists():
            return False

        # Load data
        self.ds: DatasetEntries = load_unet_calib_dataset_entries(
            path=self.unet_calib_path
        )
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 200

    @classmethod
    def dataset_name(cls) -> str:
        return "stable_diffusion_calib_unet"


class StableDiffusionCalibDatasetVae(StableDiffusionCalibDatasetBase):
    def _validate_data(self) -> bool:
        if not self.vae_calib_path.exists():
            return False

        # Load data
        self.ds: DatasetEntries = load_calib_dataset_entries(path=self.vae_calib_path)
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 200

    @classmethod
    def dataset_name(cls) -> str:
        return "stable_diffusion_calib_vae"


class StableDiffusionCalibDatasetControlNet(StableDiffusionCalibDatasetBase):
    def _validate_data(self) -> bool:
        if not self.controlnet_calib_path.exists():
            return False

        # Load data
        self.ds: DatasetEntries = load_calib_dataset_entries(
            path=self.controlnet_calib_path
        )
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 200

    @classmethod
    def dataset_name(cls) -> str:
        return "stable_diffusion_calib_controlnet"


class StableDiffusionCalibDatasetTextEncoder(StableDiffusionCalibDatasetBase):
    def _validate_data(self) -> bool:
        tokenizer = self.sd_cls.make_tokenizer()
        cond_tokens, uncond_token = load_calib_tokens(
            self.prompt_path, tokenizer, num_samples=self.num_samples
        )
        self.ds: DatasetEntries = dict(
            tokens=[t.numpy() for t in cond_tokens] + [uncond_token.numpy()]
        )
        return True

    @staticmethod
    def default_samples_per_job() -> int:
        return 200

    def _download_data(self) -> None:
        return  # No need to pre-generate anything for text encoder

    @classmethod
    def dataset_name(cls) -> str:
        return "stable_diffusion_calib_text_encoder"
