# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import whisper  # type: ignore
from scipy import special as scipy_special  # type: ignore

from qai_hub_models.models.whisper_asr.model import Whisper
from qai_hub_models.utils.model_adapters import TorchNumpyAdapter

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


class WhisperApp:
    """
    WhisperApp runs Whisper encoder and decoder to transcribe audio
    represented as mel spectrogram. It support all model variants of
    OpenAI Whisper.
    """

    def __init__(self, whisper: Whisper):
        decoder = whisper.decoder
        encoder = whisper.encoder
        self.num_decoder_blocks = whisper.num_decoder_blocks
        self.attention_dim = whisper.attention_dim

        # Wraps torch Module so it takes np ndarray as input and outputs
        if isinstance(encoder, torch.nn.Module):
            self.encoder = TorchNumpyAdapter(encoder)
        else:
            self.encoder = encoder
        if isinstance(decoder, torch.nn.Module):
            self.decoder = TorchNumpyAdapter(decoder)
        else:
            self.decoder = decoder

    def predict(self, *args, **kwargs):
        # See transcribe.
        return self.transcribe(*args, **kwargs)

    def transcribe(self, mel_input: np.ndarray) -> str:
        """
        Transcribe an audio to text.

        Parameters:

        - mel_input: of shape (1, 80, 3000). Mel spectrogram of 30s audio.

        Returns:

        - transcribed texts
        """
        cross_attn_cache = self.encoder(mel_input)
        # Start decoding
        # coreml only takes float tensors
        x = np.array([[TOKEN_SOT]])
        decoded_tokens = [TOKEN_SOT]
        cache_tensor = np.array([], dtype=np.float32).reshape(
            (1, 0, self.attention_dim)
        )
        self_attn_cache = [cache_tensor] * 2 * self.num_decoder_blocks

        sample_len = 224  # max # of tokens to sample
        sum_logprobs = 0
        for i in range(sample_len):
            decoder_out = self.decoder(x, *cross_attn_cache, *self_attn_cache)
            # logit has shape (1, decoded_len, 51864)
            logits = decoder_out[0]
            self_attn_cache = decoder_out[1:]  # type: ignore
            # logit has shape (51864,)
            logits = logits[0, -1]  # consider only the last token

            # Filters
            # SuppressBlank
            if i == 0:
                logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
            # SuppressTokens
            logits[NON_SPEECH_TOKENS] = -np.inf

            logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)

            if i == 0:
                # detect no_speech
                no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                if no_speech_prob > NO_SPEECH_THR:
                    break

            # temperature = 0
            next_token = np.argmax(logits)
            if next_token == TOKEN_EOT:
                break

            sum_logprobs += logprobs[next_token]
            x = np.array([[next_token]])
            decoded_tokens.append(int(next_token))

        tokenizer = whisper.decoding.get_tokenizer(
            multilingual=False, language="en", task="transcribe"
        )

        text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
        return text.strip()


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
    logits: np.ndarray, tokens: List[int]
) -> Tuple[np.ndarray, float]:
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


def load_audio(mel_filter: np.ndarray, audio_path: str) -> np.ndarray:
    """
    Load audio to a mel spectrogram.
    """
    with np.load(audio_path) as f:
        audio_np = f["audio"]
    # Pad 30-seconds of silence to the input audio, for slicing
    input_feature = log_mel_spectrogram(mel_filter, audio_np, pad_to_length=N_SAMPLES)
    # input_feature has fixed shape [1, 80, 3000]. 80 is
    # spectrogram feature dim, 3000 is due to Whisper only takes
    # 30 seconds input represented as 10ms spectrogram segments
    assert input_feature.shape == (1, 80, 3000)
    return input_feature


def load_mel_filter(mel_filter_path: str) -> np.ndarray:
    with np.load(mel_filter_path) as f:
        return f["mel_80"]


# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py
def log_mel_spectrogram(
    mel_filter: np.ndarray,
    audio_np: np.ndarray,
    pad_to_length: int,
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
    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = torch.from_numpy(mel_filter) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().float().numpy()
