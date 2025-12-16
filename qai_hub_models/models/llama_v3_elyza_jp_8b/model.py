# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.datasets import get_dataset_from_name
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base,
    Llama3Base_AIMETOnnx,
    Llama3Base_QNN,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.model import (
    determine_precision_from_checkpoint,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


NUM_LAYERS = 32
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 4096
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging face repo name and url
HF_REPO_NAME = "elyza/Llama-3-ELYZA-JP-8B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 150

DEFAULT_PROMPT_CONTEXT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
DEFAULT_USER_PROMPT = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {Precision.w4a16: "llama3_8b_elyza_ckpt_w4a16"}


class Llama3_Elyza_JP_8B(Llama3Base):
    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        *args,
        **kwargs,
    ):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,  # noqa: B026
            **kwargs,
        )

    def _verify_ckpt(self):
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    # Only changes the defaults
    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = DEFAULT_USER_PROMPT,
        system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    ) -> str:
        return Llama3Base.get_input_prompt_with_tags(
            user_input_prompt=user_input_prompt,
            system_context_prompt=system_context_prompt,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        _skip_optimizations: list[str] | None = None,
    ) -> Llama3_Elyza_JP_8B:
        """
        Load a pre-trained Llama 3 (8B) model from Meta via HuggingFace.

        checkpoint:
            Local path or Hugging Face name of floating point checkpoint.
        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        """
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            llm_io_type=llm_io_type,
        )


class Llama3_Elyza_JP_8B_AIMETOnnx(Llama3Base_AIMETOnnx):
    def __init__(self, checkpoint: str | os.PathLike | Path | None, *args, **kwargs):
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,  # noqa: B026
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = "DEFAULT",
        host_device: torch.device | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        fp_model: torch.nn.Module | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Llama3_Elyza_JP_8B_AIMETOnnx:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Parameters
        ----------
        - checkpoint: Path to previously calibrated AIMET encodings and ONNX
          models. Note that encodings are sensitive to AIMET ONNX versions.
          If passing None, initializes without encodings.
        """
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in SUPPORTED_PRECISIONS:
                available_precisions = [str(p) for p in SUPPORTED_PRECISIONS]
                raise ValueError(
                    f"This model is not supported for {precision!s} precision. "
                    f"Models are available in following precisions: {','.join(available_precisions)}."
                )
            if precision not in DEFAULT_CHECKPOINT:
                available_checkpoints = [str(p) for p in DEFAULT_CHECKPOINT]
                raise ValueError(
                    f"No checkpoint is available for this model in {precision!s} precision. If you would "
                    f"like to continue with this precision, please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available_checkpoints)}."
                )
            precision_checkpoint = DEFAULT_CHECKPOINT[precision]
            checkpoint = os.path.join(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, precision_checkpoint + ".zip"
                ).fetch(extract=True),
                precision_checkpoint,
            )
            # Generate necessary ONNX models
            if fp_model is not None:
                cls.create_onnx_models(
                    checkpoint=checkpoint,
                    fp_model=fp_model,
                    context_length=context_length,
                    export_sequence_lengths=[sequence_length],
                    host_device=host_device,
                    llm_io_type=fp_model.llm_io_type,
                )

                cls.save_tokenizer_and_config(checkpoint=checkpoint, fp_model=fp_model)
        return super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )

    @staticmethod
    def get_output_names():
        return Llama3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            llm_io_type=llm_io_type,
        )

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        if num_samples == 0:
            num_samples = math.ceil(84000 / self.context_length)
        eng_num_samples, ja_num_samples = (
            round(num_samples * 0.95),
            round(num_samples * 0.05),
        )
        dataset_eng = get_dataset_from_name(
            name="wikitext",
            split=DatasetSplit.TRAIN,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=eng_num_samples,
        )
        english_dataset_entries = DataLoader(
            dataset_eng,
            batch_size=1,
            collate_fn=dataset_eng.collate_fn,
        )
        dataset_ja = get_dataset_from_name(
            name="wikitext_ja",
            split=DatasetSplit.TRAIN,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=ja_num_samples,
        )
        japanese_dataset_entries = DataLoader(
            dataset_ja,
            batch_size=1,
            collate_fn=dataset_ja.collate_fn,
        )

        dataloader = itertools.chain(english_dataset_entries, japanese_dataset_entries)
        num_combined_entries = len(english_dataset_entries) + len(
            japanese_dataset_entries
        )

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        assert self.EmbeddingClass is not None
        rope_embeddings = self.EmbeddingClass(
            max_length=self.context_length, config=self.llm_config
        )
        generator = LLM_Generator([self], self.tokenizer, rope_embeddings)

        # for data in dataloader
        for sample in tqdm(
            dataloader, total=num_combined_entries, desc="Pre-filling calibration data"
        ):
            input_ids, attention_mask, _ = sample
            for prefilled_inputs in generator.prefill(input_ids, attention_mask):
                for i, tensor in enumerate(prefilled_inputs):
                    inputs[i].append(tensor)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))


class Llama3_Elyza_JP_8B_QNN(Llama3Base_QNN):
    num_layers_per_split: int = NUM_LAYERS_PER_SPLIT

    @staticmethod
    def get_output_names():
        return Llama3_Elyza_JP_8B_QNN._get_output_names(NUM_LAYERS)

    get_input_spec = staticmethod(Llama3_Elyza_JP_8B.get_input_spec)
