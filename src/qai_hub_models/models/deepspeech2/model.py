# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import qai_hub as hub
import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.common import SampleInputsType  # noqa: TID251
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

# deepspeech_pytorch (pinned at 709df90) uses mutable dataclass defaults, raises ValueError on Python 3.11+.
_ds_spec = importlib.util.find_spec("deepspeech_pytorch")
if _ds_spec and _ds_spec.origin:
    _tc = Path(_ds_spec.origin).parent / "configs" / "train_config.py"
    if "default_factory=SpectConfig" not in _tc.read_text():
        subprocess.run(
            [
                "patch",
                str(_tc),
                str(Path(__file__).parent / "patches" / "deepspeech_patches.diff"),
            ],
            check=True,
        )
        sys.modules.pop("deepspeech_pytorch.configs.train_config", None)

from deepspeech_pytorch.configs.train_config import (  # noqa: E402
    AdamConfig,
    BiDirectionalConfig,
    SpectConfig,
)
from deepspeech_pytorch.model import BatchRNN, DeepSpeech  # noqa: E402

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://github.com/SeanNaren/deepspeech.pytorch/releases/"
    "download/V3.0/librispeech_pretrained_v3.ckpt",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "librispeech_pretrained_v3.ckpt",
)
DEFAULT_AUDIO = CachedWebModelAsset.from_asset_store(
    "hf_whisper_asr_shared", 1, "audio/jfk.wav"
)

# character set: blank token, apostrophe, A - Z, space
LABELS = ["_", "'", *string.ascii_uppercase, " "]
BLANK_IDX = 0


class DeepSpeech2(BaseModel):
    """DeepSpeech2 model for speech recognition."""

    def __init__(self, model: DeepSpeech) -> None:
        super().__init__(model)
        self.model: DeepSpeech
        # Replace each BatchRNN with BatchRNNNoPack, which removes the
        # pack_padded_sequence / pad_packed_sequence calls that are not
        # supported for export.
        for i, rnn_layer in enumerate(self.model.rnns):
            replacement = BatchRNNNoPack(
                input_size=rnn_layer.input_size,
                hidden_size=rnn_layer.hidden_size,
                rnn_type=type(rnn_layer.rnn),
                bidirectional=rnn_layer.bidirectional,
                batch_norm=rnn_layer.batch_norm is not None,
            )
            replacement.load_state_dict(rnn_layer.state_dict())
            self.model.rnns[i] = replacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DeepSpeech2.

        Parameters
        ----------
        x
            Input tensor of shape (batch, time, freq).

        Returns
        -------
        torch.Tensor
            Token indices of shape (batch, time_out), produced by argmax over
            the CTC output distribution.
        """
        x = x.transpose(1, 2).unsqueeze(1)
        lengths = torch.full((x.size(0),), x.size(3), dtype=torch.int, device=x.device)

        output_lengths = _get_seq_lens(self.model, lengths)

        # Conv stack
        for module in self.model.conv.seq_module:
            x = module(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dim
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # RNN stack (BatchRNNNoPack layers, no pack/unpack)
        for rnn in self.model.rnns:
            x = rnn(x, output_lengths)

        # Optional lookahead (unidirectional models only)
        if not self.model.bidirectional:
            x = self.model.lookahead(x)

        # Fully-connected + softmax
        x = self.model.fc(x)
        x = x.transpose(0, 1)
        x = self.model.inference_softmax(x)

        return torch.argmax(cast(torch.Tensor, x), dim=-1)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """Generate sample inputs for DeepSpeech2."""
        from qai_hub_models.models.deepspeech2.app import DeepSpeech2App

        app = DeepSpeech2App(self)
        input_spec = input_spec or self.get_input_spec()
        num_frames = input_spec["input"][0][1]
        app.context_len = num_frames

        audio_path = str(DEFAULT_AUDIO.fetch())
        spec = app.preprocess_audio(audio_path)

        # Pad/crop to num_frames
        current_len = spec.shape[1]
        if current_len > num_frames:
            spec = spec[:, :num_frames, :]
        elif current_len < num_frames:
            pad_len = num_frames - current_len
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_len))

        return {"input": [spec.numpy()]}

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_frames: int = 800,
        num_features: int = 161,
    ) -> InputSpec:
        return {
            "input": TensorSpec(
                shape=(batch_size, num_frames, num_features), dtype="float32"
            )
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.float,
        other_compile_options: str = "",
        device: hub.Device | None = None,
        model_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, model_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io"
        return compile_options

    @staticmethod
    def get_output_names() -> list[str]:
        return ["indices"]

    def get_output_spec(self) -> OutputSpec:
        input_spec = self.get_input_spec()
        batch_size = input_spec["input"][0][0]
        num_frames = input_spec["input"][0][1]
        return {
            "indices": TensorSpec(
                shape=(batch_size, num_frames),
                dtype="int32",
                description="Argmax token indices over CTC output distribution.",
            )
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None) -> Self:
        """Load DeepSpeech2 with pretrained weights."""
        upstream = DeepSpeech(
            labels=LABELS,
            model_cfg=BiDirectionalConfig(),
            precision=32,
            optim_cfg=AdamConfig(),
            spect_cfg=SpectConfig(),
        )

        if checkpoint_path is None:
            checkpoint_path = str(DEFAULT_WEIGHTS.fetch())
        state_dict = _load_checkpoint(checkpoint_path)
        upstream.load_state_dict(state_dict, strict=True)
        model = cls(model=upstream)
        model.eval()
        return model


def _get_seq_lens(model: DeepSpeech, input_length: torch.Tensor) -> torch.Tensor:
    """Compute output sequence lengths after the conv layers.

    Uses int() on conv parameters so they are treated as Python constants
    during tracing rather than being traced as tensor operations.
    """
    seq_len = input_length
    for m in model.conv.modules():
        if isinstance(m, nn.modules.conv.Conv2d):
            seq_len = (
                seq_len
                + 2 * int(m.padding[1])
                - int(m.dilation[1]) * (int(m.kernel_size[1]) - 1)
                - 1
            ) // int(m.stride[1]) + 1
    return seq_len.int()


class BatchRNNNoPack(BatchRNN):
    """
    Drop-in replacement for :class:`deepspeech_pytorch.model.BatchRNN` that
    removes ``pack_padded_sequence`` / ``pad_packed_sequence`` calls so the
    model can be exported via ``torch.export``.
    """

    def forward(
        self,
        x: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            # (T x N x H*2) -> (T x N x H) by summing both directions
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2)
        return x


def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    # weights_only=False is required because the checkpoint stores non-tensor
    # objects (optimizer config, training metadata) via pickle. This file is
    # always loaded from the verified, cached DEFAULT_WEIGHTS asset.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in sd.items():
        out[k.removeprefix("model.")] = v
    return out
