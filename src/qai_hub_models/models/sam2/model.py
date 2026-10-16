# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import sam2
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from qai_hub.client import Device
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base as Sam2
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.datasets.coco import CocoSamPointDataset
from qai_hub_models.datasets.sav import SaVDataset
from qai_hub_models.models.templates.sam.app import MASK_THRESHOLD
from qai_hub_models.models.templates.sam.evaluator import SAMMaskIoUEvaluator
from qai_hub_models.models.templates.sam.utils import copy_configs
from qai_hub_models.models.templates.sam2.model import (
    SAM2Decoder as SAM2DecoderBase,
)
from qai_hub_models.models.templates.sam2.model import (
    SAM2Encoder as SAM2EncoderBase,
)
from qai_hub_models.models.templates.sam2.model import (
    SAM2Loader as SAM2LoaderBase,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

BASE_PLUS_MODEL_TYPE = "base_plus"
LARGE_MODEL_TYPE = "large"
SMALL_MODEL_TYPE = "small"
TINY_MODEL_TYPE = "tiny"
DEFAULT_MODEL_TYPE = TINY_MODEL_TYPE
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

MODEL_REGISTERY = {
    BASE_PLUS_MODEL_TYPE: "sam2.1_hiera_base_plus.pt",
    LARGE_MODEL_TYPE: "sam2.1_hiera_large.pt",
    SMALL_MODEL_TYPE: "sam2.1_hiera_small.pt",
    TINY_MODEL_TYPE: "sam2.1_hiera_tiny.pt",
}

CONFIG_REGISTERY = {
    TINY_MODEL_TYPE: "sam2.1_hiera_t",
    SMALL_MODEL_TYPE: "sam2.1_hiera_s",
    LARGE_MODEL_TYPE: "sam2.1_hiera_l",
    BASE_PLUS_MODEL_TYPE: "sam2.1_hiera_b+",
}


class SAM2Encoder(SAM2EncoderBase):
    """Exportable SAM2 encoder that can be split into several parts."""

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        return SAM2Loader.load(encoder_cls=cls, model_type=model_type)[1]


class SAM2Decoder(SAM2DecoderBase):
    """
    This SAM2Decoder is taken from the class SAM2ImagePredictor.predict from sam2.

    This removes output mask resizing. Because this requires a dynamic shape to accomplish
    in the network, it's better to do this as a postprocessing step rather than in the inference
    framework itself.
    """

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        return SAM2Loader.load(decoder_cls=cls, model_type=model_type)[2]


encoderT = TypeVar("encoderT", bound=SAM2Encoder)
decoderT = TypeVar("decoderT", bound=SAM2Decoder)


class SAM2Loader(SAM2LoaderBase):
    """Helper class for loading and preparing a HTP-compatible SAM2 model."""

    @classmethod
    def load(
        cls,
        model_type: str = SMALL_MODEL_TYPE,
        encoder_cls: type[encoderT] = SAM2Encoder,  # type: ignore[assignment]
        decoder_cls: type[decoderT] = SAM2Decoder,  # type: ignore[assignment]
    ) -> tuple[Sam2, encoderT, decoderT]:
        sam2 = cls._load_sam2(model_type)
        cls._patch_sam2_for_qnn_compatibility(sam2)
        encoder = encoder_cls(sam2)
        decoder = decoder_cls(sam2)

        return sam2, encoder, decoder

    @classmethod
    def _load_sam2(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Sam2:
        """
        Get the SAM2 described by the given model type.
        SAM2 will be patched for QNN compatibility.
        """
        model_cfg_path = "build/configs/sam2.1"
        GlobalHydra.instance().clear()
        initialize(
            config_path=str(model_cfg_path),
            job_name="sam2_inference",
            version_base=None,
        )
        config_dir = QAIHM_MODELS_ROOT / MODEL_ID / "build"
        os.makedirs(config_dir, exist_ok=True)
        copy_configs(
            Path(sam2.__file__).parent / "configs" / "sam2.1",
            config_dir,
            required_subdir="qai_hub_models/models/sam2",
        )
        if model_type not in MODEL_REGISTERY:
            raise RuntimeError(f"Weights not found for model type `{model_type}`.")

        asset = CachedWebModelAsset(
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{MODEL_REGISTERY[model_type]}",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{MODEL_REGISTERY[model_type]}",
        )
        asset.fetch()
        # The pos-enc warmup caches tensors on CUDA when available, which then
        # get baked into the exported graph as CUDA constants.
        return build_sam2(
            CONFIG_REGISTERY[model_type],
            asset.local_cache_path,
            device="cpu",
            hydra_overrides_extra=[
                "++model.image_encoder.neck.position_encoding.warmup_cache=false",
                "++model.memory_encoder.position_encoding.warmup_cache=false",
            ],
        )


class SAM2(WorkbenchModelCollection):
    def __init__(self, sam2: Sam2, encoder: SAM2Encoder, decoder: SAM2Decoder) -> None:
        super().__init__({"encoder": encoder, "decoder": decoder})
        self.sam2 = sam2
        self.encoder = encoder
        self.decoder = decoder

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return SaVDataset

    def get_evaluator(self) -> BaseEvaluator:
        return SAMMaskIoUEvaluator(MASK_THRESHOLD)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoSamPointDataset]

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_points: int = 2,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            num_points=num_points,
        )

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Self:
        return cls(*SAM2Loader.load(model_type))


class SAM2Tiny(SAM2):
    @classmethod
    def from_pretrained(cls, model_type: str = TINY_MODEL_TYPE) -> Self:
        return cls(*SAM2Loader.load(model_type))


class SAM2Small(SAM2):
    @classmethod
    def from_pretrained(cls, model_type: str = SMALL_MODEL_TYPE) -> Self:
        return cls(*SAM2Loader.load(model_type))


class SAM2BasePlus(SAM2):
    @classmethod
    def from_pretrained(cls, model_type: str = BASE_PLUS_MODEL_TYPE) -> Self:
        return cls(*SAM2Loader.load(model_type))


class SAM2Large(SAM2):
    @classmethod
    def from_pretrained(cls, model_type: str = LARGE_MODEL_TYPE) -> Self:
        return cls(*SAM2Loader.load(model_type))
