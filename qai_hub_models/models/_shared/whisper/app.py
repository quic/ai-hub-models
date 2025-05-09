# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import samplerate
import sounddevice as sd
import torch
import whisper
from scipy import special as scipy_special

from qai_hub_models.models._shared.whisper.model import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    MEL_FILTER_PATH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.model_adapters import TorchNumpyAdapter


class WhisperApp:
    """
    WhisperApp runs Whisper encoder and decoder to transcribe audio
    represented as mel spectrogram. It support all model variants of
    OpenAI Whisper.
    """

    def __init__(
        self,
        whisper_encoder: ExecutableModelProtocol,
        whisper_decoder: ExecutableModelProtocol,
        num_decoder_blocks: int,
        num_decoder_heads: int,
        attention_dim: int,
        mean_decode_len: int,
        mel_filter: np.ndarray | None = None,
        sample_rate: int = SAMPLE_RATE,
        max_audio_seconds: int = CHUNK_LENGTH,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        no_speech_threshold=1.1,
    ):
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_heads = num_decoder_heads
        self.attention_dim = attention_dim
        self.mean_decode_len = mean_decode_len

        self.mel_filter = mel_filter
        if not self.mel_filter:
            MEL_FILTER_PATH.fetch()
            with np.load(MEL_FILTER_PATH.path()) as f:
                self.mel_filter = f[f"mel_{N_MELS}"]

        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.n_fft = n_fft
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate
        self.tokenizer = whisper.decoding.get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )
        self.no_speech_threshold = no_speech_threshold
        self.clip_segment_tokens = {*self.tokenizer.special_tokens.values()}

        # Wraps models so they take np ndarray as input and outputs
        self.encoder = TorchNumpyAdapter(whisper_encoder)
        self.decoder = TorchNumpyAdapter(whisper_decoder)

    def predict(self, *args, **kwargs):
        # See transcribe.
        return self.transcribe(*args, **kwargs)

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
                print(self.tokenizer.decode(tokens), end="", flush=True)
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
                        self.tokenizer.decode(tokens[split[0] : split[1]]),
                        end="",
                        flush=True,
                    )
                if split_start != 0:
                    tokens = tokens[split_start:]

        print("Listening...")
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
        transcribed text
        """
        tokens = self.transcribe_tokens(audio, audio_sample_rate)
        return self.tokenizer.decode(tokens).strip()

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

    def _transcribe_single_chunk(self, audio: np.ndarray) -> list[int]:
        """
        Transcribe an audio chunk to text.

        Parameters:

        audio: numpy array
            A numpy array of audio of shape (number of samples).
            The sample rate of this audio must be self.sample_rate.
            The maximum length of this audio must be self.max_audio_samples.

        Returns:
        - transcribed tokens
        """
        assert self.mel_filter is not None
        mel_input = log_mel_spectrogram(
            self.mel_filter, audio, self.max_audio_samples, self.n_fft, self.hop_length
        )
        k_cache_cross, v_cache_cross = self.encoder(mel_input)
        # Start decoding
        # coreml only takes float tensors
        x = np.array([[TOKEN_SOT]])
        decoded_tokens: list[int] = []
        sample_len = self.mean_decode_len  # mean # of tokens to sample
        k_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                self.attention_dim // self.num_decoder_heads,
                sample_len,
            )
        ).astype(np.float32)
        v_cache_self = np.zeros(
            (
                self.num_decoder_blocks,
                self.num_decoder_heads,
                sample_len,
                self.attention_dim // self.num_decoder_heads,
            )
        ).astype(np.float32)

        sum_logprobs = 0
        for i in range(sample_len):
            # Using i to index inside the decoder model hurts the
            # the model performance.
            # index - used to get positional embedding correctly.
            index = torch.zeros([1, 1], dtype=torch.int32)
            index[0, 0] = i
            decoder_out = self.decoder(
                x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self
            )
            # logit has shape (1, decoded_len, 51864)
            logits = decoder_out[0]
            k_cache_self = decoder_out[1]
            v_cache_self = decoder_out[2]

            # logit has shape (51864,)
            logits = logits[0, -1]  # consider only the last token

            # Filters
            # SuppressBlank
            if i == 0:
                logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
            # SuppressTokens
            logits[NON_SPEECH_TOKENS] = -np.inf

            logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)
            assert isinstance(logprobs, np.ndarray)

            if i == 0:
                # detect no_speech
                no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                if no_speech_prob > NO_SPEECH_THR:
                    break

            # temperature = 0
            next_token = np.argmax(logits)
            if next_token == TOKEN_EOT:
                break

            sum_logprobs += np.abs(logprobs[next_token])
            x = np.array([[next_token]])
            decoded_tokens.append(int(next_token))

        if (sum_logprobs / len(decoded_tokens)) >= self.no_speech_threshold:
            return []

        return decoded_tokens


# Whisper constants
TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L600
NON_SPEECH_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50357,
    50358,
    50359,
    50360,
    50361,
]

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L545
precision = 0.02  # in second
max_initial_timestamp = 1.0  # in second
max_initial_timestamp_index = int(max_initial_timestamp / precision)


def apply_timestamp_rules(
    logits: np.ndarray, tokens: list[int]
) -> tuple[np.ndarray, float | np.ndarray]:
    """
    When predicting timestamps, there are a few post processing rules /
    heuristics to ensure well-formed timestamps. See in-line comments for details

    Args:
    - logits: of shape (51864,)

    Returns:

    - modified logits
    - log probability of modified logits (log(softmax(logits)))
    """
    # Require producing timestamp
    logits[TOKEN_NO_TIMESTAMP] = -np.inf

    # timestamps have to appear in pairs, except directly before EOT
    seq = tokens[SAMPLE_BEGIN:]
    last_was_timestamp = len(seq) >= 1 and seq[-1] >= TOKEN_TIMESTAMP_BEGIN
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= TOKEN_TIMESTAMP_BEGIN
    if last_was_timestamp:
        if penultimate_was_timestamp:  # has to be non-timestamp
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
        else:  # cannot be normal text tokens
            logits[:TOKEN_EOT] = -np.inf

    timestamps = [t for t in tokens if t >= TOKEN_TIMESTAMP_BEGIN]
    if len(timestamps) > 0:
        # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
        # also force each segment to have a nonzero length, to   prevent infinite looping
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        logits[TOKEN_TIMESTAMP_BEGIN:timestamp_last] = -np.inf

    if len(tokens) == SAMPLE_BEGIN:
        # suppress generating non-timestamp tokens at the beginning
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

        # apply the `max_initial_timestamp` option
        last_allowed = TOKEN_TIMESTAMP_BEGIN + max_initial_timestamp_index
        logits[(last_allowed + 1) :] = -np.inf

    # if sum of probability over timestamps is above any other token, sample timestamp
    logprobs = scipy_special.log_softmax(logits)
    timestamp_logprob = scipy_special.logsumexp(logprobs[TOKEN_TIMESTAMP_BEGIN:])
    max_text_token_logprob = logprobs[:TOKEN_TIMESTAMP_BEGIN].max()
    if timestamp_logprob > max_text_token_logprob:
        # Mask out all but timestamp tokens
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

    return logits, logprobs


# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py
def log_mel_spectrogram(
    mel_filter: np.ndarray,
    audio_np: np.ndarray,
    pad_to_length: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio_np: np.ndarray, shape = (*)

    pad_to_length: int
        Add zero samples to the right till this length. No op if
        len(audio) >= pad_to_length

    Returns
    -------
    np.ndarray, shape = (1, 80, n_frames)
        A Tensor that contains the Mel spectrogram. n_frames = 3000 for whisper
    """
    audio = torch.from_numpy(audio_np)
    assert isinstance(audio, torch.Tensor)

    if pad_to_length is not None:
        padding = pad_to_length - len(audio)
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = torch.from_numpy(mel_filter) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().float().numpy()


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
