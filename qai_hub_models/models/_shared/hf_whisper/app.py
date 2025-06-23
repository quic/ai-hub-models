# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import samplerate
import torch

from qai_hub_models.models._shared.hf_whisper.model import (
    CHUNK_LENGTH,
    MEAN_DECODE_LEN,
    SAMPLE_RATE,
    HfWhisper,
    get_feature_extractor,
    get_tokenizer,
)


class HfWhisperApp:
    """
    HfWhisperApp runs Whisper encoder and decoder to transcribe audio
    represented as mel spectrogram. It support all model variants of
    OpenAI Whisper in huggingface.
    """

    def __init__(
        self,
        hf_whisper: HfWhisper,
        sample_rate: int = SAMPLE_RATE,
        max_audio_seconds: int = CHUNK_LENGTH,
    ):
        self.decoder = hf_whisper.decoder.to("cpu").eval()
        self.encoder = hf_whisper.encoder.to("cpu").eval()
        self.config = hf_whisper.config

        self.mean_decode_len = MEAN_DECODE_LEN

        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate

        self.feature_extractor = get_feature_extractor(hf_whisper.hf_source)
        self.tokenizer = get_tokenizer(hf_whisper.hf_source)

    def predict(self, *args, **kwargs):
        # See transcribe.
        return self.transcribe(*args, **kwargs)

    def transcribe(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> str:
        """
        Transcribe the provided audio to text.

        Parameters
        ----------
        audio: numpy array | str
            Path to audio file if a string.
            Raw audio array of shape (# of samples) if a numpy array.

        audio_sample_rate: int | None
            The sample rate of the provided audio, in samples / second.
            If audio is a numpy array, this must be provided.
            If audio is a file and audio_sample_rate is None, this is ignored and the sample rate will be derived from the audio file.

        Returns
        -------
        List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
        """
        if isinstance(audio, str):
            import audio2numpy as a2n  # import here, as this requires ffmpeg to be installed on host machine

            audio, audio_sample_rate = a2n.audio_from_file(audio)
        else:
            assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)
        assert isinstance(audio_sample_rate, int)
        with torch.no_grad():
            trans = " ".join(
                self._transcribe_single_chunk(x)
                for x in chunk_and_resample_audio(audio, audio_sample_rate)
            )

        return trans

    def _transcribe_single_chunk(self, audio: np.ndarray) -> str:
        """
        Transcribe an audio chunk to text.

        Parameters:

        audio: numpy array
            A numpy array of audio of shape (number of samples).
            The sample rate of this audio must be self.sample_rate.
            The maximum length of this audio must be self.max_audio_samples.

        Returns:

        - transcribed texts
        """
        # feature
        input_features = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )["input_features"]

        # encoder
        kv_cache_cross = self.encoder(input_features)
        if not isinstance(kv_cache_cross, tuple):
            kv_cache_cross = (kv_cache_cross,)

        sot = self.config.decoder_start_token_id
        num_decoder_blocks = self.config.decoder_layers
        attention_dim = self.config.d_model
        num_decoder_heads = self.config.decoder_attention_heads
        mask_neg = self.config.mask_neg
        eot = self.config.eos_token_id

        # decoder
        output_ids = torch.tensor([[sot]])  # Start of transcript
        output_logits = []
        output_length = output_ids.shape[1]

        position_ids = torch.tensor([0], dtype=torch.int32)
        attention_mask = torch.full(
            (1, 1, 1, self.mean_decode_len),
            mask_neg,
            dtype=torch.float32,
        )

        # init kv_cache_self
        k_cache_self = torch.zeros(
            (
                num_decoder_heads,
                1,
                attention_dim // num_decoder_heads,
                self.mean_decode_len - 1,
            ),
            dtype=torch.float32,
        )
        v_cache_self = torch.zeros(
            (
                num_decoder_heads,
                1,
                self.mean_decode_len - 1,
                attention_dim // num_decoder_heads,
            ),
            dtype=torch.float32,
        )
        kv_cache_self = tuple(
            (k_cache_self, v_cache_self) for _ in range(num_decoder_blocks)
        )

        for n in range(self.mean_decode_len - 1):
            # get current token
            input_ids = output_ids[:, n : n + 1].to(torch.int32)

            # update attention_mask
            attention_mask[:, :, :, self.mean_decode_len - n - 1] = 0.0

            # flattened kv caches input
            flattened_kv_cache_self = tuple(
                item for sublist in kv_cache_self for item in sublist
            )
            flattened_kv_cache_cross = tuple(
                item for sublist in kv_cache_cross for item in sublist
            )

            # decode and update kv_cache_self
            decoder_input = (
                (input_ids, attention_mask)
                + flattened_kv_cache_self
                + flattened_kv_cache_cross
                + (position_ids,)
            )

            decoder_output = self.decoder(*decoder_input)
            if isinstance(decoder_output, tuple):
                logits, kv_cache_self = decoder_output
            else:
                logits = decoder_output[0]
                kv_cache_self = tuple(
                    decoder_output[i : i + 2] for i in range(1, len(decoder_output), 2)
                )

            # update output_logits
            output_logits.append(logits.detach().clone())

            # update output_ids
            output_id = torch.argmax(logits, 1).squeeze(0)
            # end of transcript
            if len(output_logits) == (self.mean_decode_len - 1) or output_id == eot:
                output_ids = torch.cat((output_ids, output_id), -1)
                break
            if n >= output_length - 1:
                output_ids = torch.cat((output_ids, output_id), -1)

            # update position_ids
            position_ids += 1

        # Exclude start / end tokens
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> list[np.ndarray]:
    """
    Parameters
    ----------
    audio: str
        Raw audio numpy array of shape [# of samples]

    audio_sample_rate: int
        Sample rate of audio array, in samples / sec.

    model_sample_rate: int
        Sample rate (samples / sec) required to run Whisper. The audio file
        will be resampled to use this rate.

    model_chunk_seconds: int
        Split the audio in to N sequences of this many seconds.
        The final split may be shorter than this many seconds.

    Returns
    -------
    List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
    """
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    number_of_full_length_audio_chunks = (
        audio.shape[0] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = (
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )

    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
        ),
        audio[last_sample_in_full_length_audio_chunks:],
    ]
