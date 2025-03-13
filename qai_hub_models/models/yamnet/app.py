# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import resampy
import soundfile as sf
import torch

from qai_hub_models.models.yamnet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YAMNET_PROXY_REPO_COMMIT,
    YAMNET_PROXY_REPOSITORY,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

SAMPLE_RATE = 16000
CHUNK_LENGTH = 0.98


def preprocessing_yamnet_from_source(waveform_for_torch: torch.Tensor):
    """
    Args:
        waveform (torch.Tensor): Tensor of audio of dimension (..., time)

    Returns:
        patches : batched torch tsr of shape [N, C, T]
        spectrogram :  Mel frequency spectrogram of size (..., ``n_mels``, time)
    """

    with SourceAsRoot(
        YAMNET_PROXY_REPOSITORY,
        YAMNET_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        from torch_audioset.data.torch_input_processing import (
            WaveformToInput as TorchTransform,
        )

        #  This is a _log_ mel-spectrogram transform that adheres to the transform
        #  used by Google's vggish model input processing pipeline
        patches, spectrogram = TorchTransform().wavform_to_log_mel(
            waveform_for_torch, SAMPLE_RATE
        )

    return patches, spectrogram


def parse_category_meta():

    """Read the class name definition file and return a list of strings."""
    YAMNET_CLASS_CSV = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "yamnet_class_map.csv"
    )
    accu = []
    with open(YAMNET_CLASS_CSV.fetch()) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        for (inx, category_id, category_name) in reader:
            accu.append(category_name)
    return accu


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
        Sample rate (samples / sec) required to run Yamnet. The audio file
        will be resampled to use this rate.

    model_chunk_seconds: int
        Split the audio in to N sequences of this many seconds.
        The final split may be shorter than this many seconds.

    Returns
    -------
    List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
    """
    if audio_sample_rate != model_sample_rate:
        audio = resampy.resample(audio, audio_sample_rate, model_sample_rate)
        audio_sample_rate = model_sample_rate
    number_of_full_length_audio_chunks = int(
        audio.shape[1] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = int(
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )
    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:, :last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
            axis=1,
        ),
    ]


def load_audiofile(path: str | Path):
    """
    Decode the WAV file.
        Parameters:
            path: Path of the input audio.

        Returns:
            x: Reads audio sample from path and converts to torch tensor.
            sr : sampling rate of audio samples

    """
    x, sr = sf.read(path, dtype="int16", always_2d=True)
    x = x / 2**15
    x = x.T.astype(np.float32)
    # Convert to mono and the sample rate expected by YAMNet.
    if x.shape[0] > 1:
        x = np.mean(x, axis=1)
    return x, sr


class YamNetApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an YamNetApp.

    For a given audio input, the app will:
        * Pre-process the audio
        * Run audio Classification
        * Return the class index of audio CHUNK_LENGTH of 0.98 seconds.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, path: str | Path, audio_sample_rate: int | None = None):
        """
        Predict audio classes for given sample.

        Parameters
        ----------
        path: numpy array | str
            Path to audio file if a string.

        audio_sample_rate: int | None
            The sample rate of the provided audio, in samples / second.
            If audio is a numpy array, this must be provided.
            If audio is a file and audio_sample_rate is None, this is ignored and the sample rate will be derived from the audio file.

        Returns
        -------
        List of class ids from AudioSet-YouTube corpus is returned.
        """

        audio, audio_sample_rate = load_audiofile(path)

        assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)
        accu = []
        for x in chunk_and_resample_audio(audio, audio_sample_rate):
            pred = self.classify(x)
            accu.append(pred)
        accu = np.stack(accu)
        # Average them along time to get an overall classifier output for the clip.
        mean_scores = np.mean(accu, axis=0)
        mean_scores = mean_scores[0]
        top_N = 5
        # Report the highest-scoring classes.
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        # Label each audio one-by-one, for all the chunks,
        actions = parse_category_meta()
        return [actions[prediction] for prediction in top_class_indices]

    def classify(self, segment: np.ndarray) -> np.ndarray:
        """
        From the provided audio samples,calculate scores(matrix of
        time_frames, num_classes classifier scores).

        Parameters:
            segment: chunked audio samples

        Returns:
            raw_prediction: class_probs for each chunk of audio samples
        """
        segment = torch.tensor(segment)
        patches, spectrogram = preprocessing_yamnet_from_source(segment)
        # Inference using mdoel
        raw_prediction = self.model(patches)
        raw_prediction = raw_prediction.numpy()
        return raw_prediction
