# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from transformers.models.whisper import WhisperConfig

from qai_hub_models.models._shared.hf_whisper.model import (
    CHUNK_LENGTH,
    MASK_NEG,
    MEAN_DECODE_LEN,
    SAMPLE_RATE,
    get_feature_extractor,
    get_tokenizer,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol


class HfWhisperApp:
    """
    HfWhisperApp runs Whisper encoder and decoder to transcribe audio
    represented as mel spectrogram. It support all model variants of
    OpenAI Whisper in huggingface.
    """

    def __init__(
        self,
        encoder: ExecutableModelProtocol,
        decoder: ExecutableModelProtocol,
        hf_model_id: str,
        sample_rate: int = SAMPLE_RATE,
        max_audio_seconds: int = CHUNK_LENGTH,
    ):
        self.encoder = encoder
        self.decoder = decoder
        if isinstance(self.decoder, torch.nn.Module):
            self.decoder = self.decoder.to("cpu").eval()
        if isinstance(self.encoder, torch.nn.Module):
            self.encoder = self.encoder.to("cpu").eval()

        self.config = WhisperConfig.from_pretrained(hf_model_id)
        self.config.return_dict = False
        self.config.tie_word_embeddings = False
        self.config.mask_neg = MASK_NEG

        self.mean_decode_len = MEAN_DECODE_LEN

        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate

        self.feature_extractor = get_feature_extractor(hf_model_id)
        self.tokenizer = get_tokenizer(hf_model_id)
        self.clip_segment_tokens = set(self.tokenizer.all_special_ids)

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
        tokens = self.transcribe_tokens(audio, audio_sample_rate)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _transcribe_single_chunk(self, audio: np.ndarray) -> list[int]:
        """
        Transcribe an audio chunk to text.

        Parameters:

        audio: numpy array
            A numpy array of audio of shape (number of samples).
            The sample rate of this audio must be self.sample_rate.
            The maximum length of this audio must be self.max_audio_samples.

        Returns:
            list of token ids
        """
        # feature
        input_features = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )["input_features"]

        # encoder
        kv_cache_cross = self.encoder(input_features)
        if not isinstance(kv_cache_cross, tuple):
            kv_cache_cross = (kv_cache_cross,)
        if not isinstance(kv_cache_cross[0], (tuple, list)):
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
            if isinstance(decoder_output, tuple) and len(decoder_output) == 2:
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

        return output_ids[0].tolist()

    def stream(self, device=2, audio_chunk_size_seconds: int = 5) -> None:
        """
        Stream audio from the given audio device and transcribe in real time.

        Parameters:
            device:
                Audio device (see. sounddevice.query_devices())
            audio_chunk_size_seconds:
                Number of seconds to record between each transcription attempt.
        """
        tokens: list[int] = []

        def callback(audio: np.ndarray, frames, time, status):
            nonlocal tokens
            curr_tokens = self.transcribe_tokens(audio.squeeze(-1), SAMPLE_RATE)
            tokens.extend(curr_tokens)

            if not curr_tokens:
                # This audio was empty, so it's safe to decode previous tokens.
                print(
                    self.tokenizer.decode(tokens, skip_special_tokens=True),
                    end="",
                    flush=True,
                )
                tokens = []
            else:
                split_start = 0
                decode_splits = []
                token_idx = 0
                # Every time 2 "clip segment tokens" (timestamp tokens)
                # appear in sequence, we're safe to decode the previous tokens.
                while token_idx < len(tokens):
                    if tokens[token_idx] in self.clip_segment_tokens:
                        next_non_clip_idx = token_idx + 1
                        while (
                            next_non_clip_idx < len(tokens)
                            and tokens[next_non_clip_idx] in self.clip_segment_tokens
                        ):
                            next_non_clip_idx = next_non_clip_idx + 1

                        if next_non_clip_idx >= token_idx + 2:
                            split_end = token_idx + 1
                            if max(split_end - split_start, 0) > 0:
                                decode_splits.append((split_start, split_end))
                            split_start = next_non_clip_idx

                        token_idx = next_non_clip_idx + 1
                    else:
                        token_idx = token_idx + 1

                for split in decode_splits:
                    print(
                        self.tokenizer.decode(
                            tokens[split[0] : split[1]], skip_special_tokens=True
                        ),
                        end="",
                        flush=True,
                    )
                if split_start != 0:
                    tokens = tokens[split_start:]

        print("Listening...")
        print("Text can take up to 20 seconds before printing.")
        with sd.InputStream(
            device=device,
            channels=1,
            blocksize=audio_chunk_size_seconds * SAMPLE_RATE,
            callback=callback,
            samplerate=SAMPLE_RATE,
        ):
            while True:
                response = input("Press ctrl+c or q/Q to quit.\n")
                if response in ("q", "Q"):
                    break

    def transcribe_tokens(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> list[int]:
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
        transcribed tokens
        """
        if isinstance(audio, str):
            import audio2numpy as a2n  # import here, as this requires ffmpeg to be installed on host machine

            audio, audio_sample_rate = a2n.audio_from_file(audio)
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                # Audio is multi-channel (e.g., stero); collapse to single.
                audio = audio.mean(-1)

        assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)

        out_chunked_tokens: list[list[int]] = [
            self._transcribe_single_chunk(x)
            for x in chunk_and_resample_audio(audio, audio_sample_rate)
        ]
        out_tokens: list[int] = []
        for chunk_tokens in out_chunked_tokens:
            out_tokens.extend(chunk_tokens)
        return out_tokens


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
        audio = resample_poly(audio, model_sample_rate, audio_sample_rate)
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
