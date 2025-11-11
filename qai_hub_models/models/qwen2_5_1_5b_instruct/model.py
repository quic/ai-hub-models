# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import torch

from qai_hub_models.models._shared.llm.model import (
    LLMIOType,
    determine_precision_from_checkpoint,
)
from qai_hub_models.models._shared.qwen2.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Qwen2Base,
    Qwen2Base_AIMETOnnx,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 11
HIDDEN_SIZE = 1536
NUM_KEY_VALUE_HEADS = 2
NUM_ATTN_HEADS = 12

# Hugging face repo name and url
HF_REPO_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
MIN_MEMORY_RECOMMENDED = 60
DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4]
DEFAULT_CHECKPOINT = {
    Precision.w4: "qwen2515_ckpt_w4",
}


class Qwen2_5_1_5B(Qwen2Base):
    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    def __init__(
        self, checkpoint: str | os.PathLike | Path = HF_REPO_NAME, *args, **kwargs
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

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        _skip_optimizations: list[str] | None = None,
    ) -> Qwen2_5_1_5B:
        """
        Load a pre-trained Qwen 2.5 (1.5B) model via HuggingFace.

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
        _skip_optimizations:
        """
        # Since we multiply the attention mask for Qwen, the default value has
        # issues so we use the Genie value for the unquantized variant too.
        attention_mask_min_clip = -1000.0

        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            attention_mask_min_clip=attention_mask_min_clip,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names():
        return Qwen2Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Qwen2Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            llm_io_type=llm_io_type,
        )


class Qwen2_5_1_5B_AIMETOnnx(Qwen2Base_AIMETOnnx):
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
    ) -> Qwen2_5_1_5B_AIMETOnnx:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Args:

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
        return Qwen2_5_1_5B_AIMETOnnx._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Qwen2Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            llm_io_type=llm_io_type,
        )
