# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from collections.abc import Callable
from typing import Any

import k2
import kaldi_native_fbank as knf
import numpy as np
import soundfile
import torch
from qai_hub.client import DatasetEntries
from torch import Tensor
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.models.zipformer.model import HfZipformer
from qai_hub_models.utils.base_app import CollectionAppProtocol
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


def create_streaming_feature_extractor() -> knf.OnlineFbank:
    """Create a CPU streaming feature extractor.

    Returns
    -------
    OnlineFbank:
        a CPU streaming feature extractor.
    """
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return knf.OnlineFbank(opts)


def streaming_greedy_search(
    model_wrapper: HfZipformer, model_fpm: dict, features: Tensor
) -> str:
    # Return the text corresponding to audio features.
    token_file = os.path.join(os.path.dirname(__file__), "tokens.txt")
    max_frames = features.shape[1]
    segment = model_wrapper.segment
    offset = model_wrapper.offset
    blank_id = model_wrapper.blank_id
    context_size = model_wrapper.context_size
    hyp = [blank_id] * context_size
    decoder_input = torch.tensor([hyp], dtype=torch.int32)
    decoder_out = model_fpm["decoder"](decoder_input)
    state = model_wrapper.get_init_state()
    for start in range(0, max_frames, offset):
        streaming_feature = features[:, start : start + segment, :]
        if start + segment > max_frames:
            pad = start + segment - max_frames
            streaming_feature = torch.nn.functional.pad(
                streaming_feature, pad=(0, 0, 0, pad), mode="constant", value=0
            )
        encoder_out, state = model_fpm["encoder"](*[streaming_feature, *state])
        encoder_out = encoder_out.squeeze()
        for t in range(encoder_out.size(0)):
            cur_encoder_out = encoder_out[t : t + 1]
            joiner_out = model_fpm["joiner"](cur_encoder_out, decoder_out).squeeze(0)
            y = joiner_out.argmax(dim=0).item()

            if y != blank_id:
                hyp.append(y)
                decoder_input = torch.tensor([hyp[-context_size:]], dtype=torch.int32)
                decoder_out = model_fpm["decoder"](decoder_input)

    token_table = k2.SymbolTable.from_file(token_file)
    text = "".join([token_table[i] for i in hyp[2:]])
    return text.replace("▁", " ").strip()


class ZipformerApp(CollectionAppProtocol):
    """
    ZipformerApp runs Zipformer encoder, decoder and
        joiner to transcribe audio.
    """

    def __init__(
        self,
        encoder: Callable[
            [Tensor],
            tuple[torch.Tensor, tuple[Tensor, ...]],
        ],
        decoder: Callable[
            [Tensor],
            Tensor,
        ],
        joiner: Callable[
            [Tensor, Tensor],
            Tensor,
        ],
        model: HfZipformer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> str:
        return self.transcribe(*args, **kwargs)

    def transcribe(
        self, audio: np.ndarray, audio_sample_rate: int | None = None
    ) -> str:
        """
        Transcribe the provided audio to text.

        Parameters
        ----------
        audio
            Raw audio array of shape (# of samples)
        audio_sample_rate
            The sample rate of the provided audio, in samples / second.
            If audio is a numpy array, this must be provided.
            If audio is a file and audio_sample_rate is None, this is ignored and the sample rate will be derived from the audio file.

        Returns
        -------
        text
            transcribed text
        """
        online_fbank = create_streaming_feature_extractor()  # 100 frames / s
        online_fbank.accept_waveform(sampling_rate=audio_sample_rate, waveform=audio)
        frames_ = np.array(
            [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
        )
        frames = torch.tensor(frames_, dtype=torch.float32).unsqueeze(0)  # (1, len, 80)
        origin_fpm = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "joiner": self.joiner,
        }
        return streaming_greedy_search(self.model, origin_fpm, frames)

    @classmethod
    def get_calibration_data(
        cls,
        model: BaseModel,
        calibration_dataset_name: str,
        num_samples: int | None,
        input_spec: InputSpec,
        collection_model: HfZipformer,
    ) -> DatasetEntries:
        encoder_fpm = collection_model.components["ZipformerEncoder"]
        decoder_fpm = collection_model.components["ZipformerDecoder"]
        joiner_fpm = collection_model.components["ZipformerJoiner"]
        batch_size = get_batch_size(input_spec) or 1
        assert callable(encoder_fpm) and callable(decoder_fpm) and callable(joiner_fpm)
        assert batch_size == 1, batch_size
        segment = collection_model.segment
        offset = collection_model.offset
        blank_id = collection_model.blank_id
        context_size = collection_model.context_size
        dataset = get_dataset_from_name(calibration_dataset_name, DatasetSplit.TRAIN)
        num_samples = num_samples or dataset.default_samples_per_job()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]
        n = 0
        for wav_file, _ in dataloader:
            n = n + 1
            if n % 100 == 0:
                print(f"Processed {n} samples")
            wav_file = wav_file[0]
            samples, sample_rate = soundfile.read(wav_file)
            online_fbank = create_streaming_feature_extractor()
            online_fbank.accept_waveform(sample_rate, samples)

            frames_ = np.array(
                [
                    online_fbank.get_frame(i)
                    for i in range(online_fbank.num_frames_ready)
                ]
            )
            feature_array = torch.tensor(frames_, dtype=torch.float32).unsqueeze(0)
            max_frames = feature_array.shape[1]
            state = collection_model.get_init_state()
            hyp = [blank_id] * context_size
            decoder_input = torch.tensor([hyp], dtype=torch.int32)
            decoder_out = decoder_fpm(decoder_input)
            if model._get_name() == "ZipformerEncoder":
                for start in range(0, max_frames, offset):
                    streaming_feature = feature_array[:, start : start + segment, :]
                    if start + segment > max_frames:
                        pad = start + segment - max_frames
                        streaming_feature = torch.nn.functional.pad(
                            streaming_feature,
                            pad=(0, 0, 0, pad),
                            mode="constant",
                            value=0,
                        )
                    inputs[0].append(streaming_feature)
                    for i, tensor in enumerate(state):
                        inputs[i + 1].append(tensor)
                    encoder_out, state = encoder_fpm(*[streaming_feature, *state])
                    encoder_out = encoder_out.squeeze()
                    for t in range(encoder_out.size(0)):
                        cur_encoder_out = encoder_out[t : t + 1]
                        joiner_out = joiner_fpm(cur_encoder_out, decoder_out).squeeze(0)
                        y = joiner_out.argmax(dim=0).item()
                        if y != blank_id:
                            hyp.append(y)
                            decoder_input_ = hyp[-context_size:]
                            decoder_input = torch.tensor(
                                [decoder_input_], dtype=torch.int32
                            )
                            decoder_out = decoder_fpm(decoder_input)

            elif model._get_name() == "ZipformerDecoder":
                for start in range(0, max_frames, offset):
                    streaming_feature = feature_array[:, start : start + segment, :]
                    if start + segment > max_frames:
                        pad = start + segment - max_frames
                        streaming_feature = torch.nn.functional.pad(
                            streaming_feature,
                            pad=(0, 0, 0, pad),
                            mode="constant",
                            value=0,
                        )

                    encoder_out, state = encoder_fpm(*[streaming_feature, *state])
                    encoder_out = encoder_out.squeeze()
                    for t in range(encoder_out.size(0)):
                        cur_encoder_out = encoder_out[t : t + 1]
                        joiner_out = joiner_fpm(cur_encoder_out, decoder_out).squeeze(0)
                        y = joiner_out.argmax(dim=0).item()
                        inputs[0].append(decoder_input.numpy())
                        if y != blank_id:
                            hyp.append(y)
                            decoder_input_ = hyp[-context_size:]
                            decoder_input = torch.tensor(
                                [decoder_input_], dtype=torch.int32
                            )
                            decoder_out = decoder_fpm(decoder_input)

            elif model._get_name() == "ZipformerJoiner":
                for start in range(0, max_frames, offset):
                    streaming_feature = feature_array[:, start : start + segment, :]
                    if start + segment > max_frames:
                        pad = start + segment - max_frames
                        streaming_feature = torch.nn.functional.pad(
                            streaming_feature,
                            pad=(0, 0, 0, pad),
                            mode="constant",
                            value=0,
                        )

                    encoder_out, state = encoder_fpm(*[streaming_feature, *state])
                    encoder_out = encoder_out.squeeze()
                    for t in range(encoder_out.size(0)):
                        cur_encoder_out = encoder_out[t : t + 1]
                        joiner_out = joiner_fpm(cur_encoder_out, decoder_out).squeeze(0)
                        inputs[0].append(cur_encoder_out)
                        inputs[1].append(decoder_out)

                        y = joiner_out.argmax(dim=0).item()
                        if y != blank_id:
                            hyp.append(y)
                            decoder_input_ = hyp[-context_size:]
                            decoder_input = torch.tensor(
                                [decoder_input_], dtype=torch.int32
                            )
                            decoder_out = decoder_fpm(decoder_input)
            else:
                raise NotImplementedError(model._get_name())

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
