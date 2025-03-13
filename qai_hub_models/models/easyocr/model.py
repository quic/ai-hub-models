# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from easyocr.craft import CRAFT as CRAFTDetector
from easyocr.DBNet.DBNet import DBNet as DBNetDetector
from easyocr.easyocr import Reader
from easyocr.model.model import Model as Recognizer
from easyocr.model.vgg_model import Model as VGGRecognizer

from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
LANG_LIST = ["en"]
MODEL_STORAGE_DIRECTORY = None
USER_NETWORK_DIRECTORY = None


class EasyOCR(CollectionModel):
    def __init__(
        self,
        detector: EasyOCRDetector,
        recognizer: EasyOCRRecognizer,
        lang_list: list[str],
    ):
        self.lang_list = lang_list
        self.detector = detector
        self.recognizer = recognizer

    @classmethod
    def from_pretrained(
        cls,
        lang_list: list[str] = LANG_LIST,
        detect_network: str = "craft",
        recog_network: str = "standard",
    ) -> EasyOCR:
        """
        Create an EasyOCR model.

        Parameters:
            lang_list: list[str]
                Language List

            detect_network: Detector network architecture
                Valid options: craft, dbnet18

            recog_network: Recognizer network architecture
                Valid options: standard (determine based on language list), generation1, generation2
        """
        detector = EasyOCRDetector.from_pretrained(lang_list, detect_network)
        recognizer = EasyOCRRecognizer.from_pretrained(lang_list, recog_network)
        return EasyOCR(detector, recognizer, lang_list)


class EasyOCRDetector(BaseModel):
    def __init__(self, detector_model: CRAFTDetector | DBNetDetector) -> None:
        super().__init__()
        self.model = detector_model

    @classmethod
    def from_pretrained(
        cls, lang_list: list = LANG_LIST, detect_network: str = "craft"
    ) -> EasyOCRDetector:
        from easyocr.easyocr import Reader

        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
            detect_network=detect_network,
            model_storage_directory=MODEL_STORAGE_DIRECTORY,
            user_network_directory=USER_NETWORK_DIRECTORY,
        )

        return cls(ocr_reader.detector)  # pyright: ignore[reportArgumentType]

    def forward(self, image: torch.Tensor):
        """
        Run detector on `image`, and produce a ((text score map, link score map), features).

        Parameters:
            image: Pixel values pre-processed for detector consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR
        """
        if isinstance(self.model, CRAFTDetector):
            return self.model(image)
        elif isinstance(self.model, DBNetDetector):
            assert self.model.model is not None
            return self.model.model(image, training=False)
        else:
            raise NotImplementedError("Unknown detector model")

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 608,
        width: int = 800,
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names(*args, **kwargs):
        return ["results", "features"]


class EasyOCRRecognizer(BaseModel):
    """VGG based model is the default model of text recognition"""

    def __init__(self, recognizer_model: VGGRecognizer | Recognizer) -> None:
        super().__init__()
        self.model = recognizer_model

    @classmethod
    def from_pretrained(
        cls, lang_list: list = LANG_LIST, recog_network: str = "standard"
    ) -> EasyOCRRecognizer:
        ocr_reader = Reader(
            lang_list,
            gpu=False,
            quantize=False,
            recog_network=recog_network,
            model_storage_directory=MODEL_STORAGE_DIRECTORY,
            user_network_directory=USER_NETWORK_DIRECTORY,
        )

        return cls(ocr_reader.recognizer)  # pyright: ignore[reportArgumentType]

    def forward(self, image: torch.Tensor):
        """
        Run recognizer on `image`, and produce a score matrix corresponding to character classes.

        Parameters:
            image: Pixel values pre-processed for detector consumption.
                   Range: float[0, 1]
                   1-channel Color Space: grey

        Returns:
            segmented mask per class: Shape [batch, T, classes]
        """
        return self.model(image, None)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 64,
        width: int = 1000,
    ) -> InputSpec:
        return {
            "image": ((batch_size, 1, height, width), "float32"),
        }

    @staticmethod
    def get_output_names(*args, **kwargs):
        return ["output_preds"]
