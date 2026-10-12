# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""AMI Corpus datasets for pyannote speaker diarization calibration.

License: CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
Source:  https://groups.inf.ed.ac.uk/ami/corpus/

3 recordings (~102 MB) from the AMI IHM test.mini split, publicly available
under CC BY 4.0 without registration.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

_VERSION = 1
_SAMPLE_RATE = 16000
_SEG_FRAMES = 80000  # 5 s x 16 kHz
_SEG_STEP_FRAMES = 16000  # 1 s step — matches notebook seg_step_ratio=0.2 (step=1s)

# AMI IHM mix-headset recordings: test.mini split (IS1009a, ES2004a, TS3003a).
# Source: https://groups.inf.ed.ac.uk/ami/corpus/ (CC BY 4.0 license)
_AMI_IHM_FILENAMES = [
    "IS1009a.Mix-Headset.wav",
    "ES2004a.Mix-Headset.wav",
    "TS3003a.Mix-Headset.wav",
]

# Keep "ami_diarization" as the cache folder name to preserve existing cached data.
_AMI_ASSETS = [
    CachedWebDatasetAsset.from_asset_store("ami_diarization", _VERSION, filename)
    for filename in _AMI_IHM_FILENAMES
]

# RTTM reference annotations for DER evaluation.
# Source: https://github.com/pyannote/AMI-diarization-setup (Apache-2.0 license)
_AMI_RTTM_FILENAMES = ["IS1009a.rttm", "ES2004a.rttm", "TS3003a.rttm"]
_AMI_RTTM_ASSETS = [
    CachedWebDatasetAsset.from_asset_store("ami_diarization", _VERSION, filename)
    for filename in _AMI_RTTM_FILENAMES
]

# Fixed byte-length for path encoding in AMIDiarizationEvalDataset tensors.
_PATH_TENSOR_LEN = 512


def _read_wav_mono_16k(path: Path) -> np.ndarray:
    """Read a WAV file and return a float32 mono array at 16 kHz."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype: type
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_val = float(np.iinfo(dtype).max)
    samples /= max_val
    return samples


def _sliding_chunks(
    audio: np.ndarray, chunk_frames: int, step_frames: int
) -> list[np.ndarray]:
    """Slice audio into overlapping chunks, zero-padding the last partial chunk."""
    chunks = []
    start = 0
    while start + chunk_frames <= len(audio):
        chunks.append(audio[start : start + chunk_frames])
        start += step_frames
    if start < len(audio):
        tail = audio[start:]
        pad = np.zeros(chunk_frames - len(tail), dtype=audio.dtype)
        chunks.append(np.concatenate([tail, pad]))
    return chunks


def _apply_speaker_masks(
    fbank_chunk: torch.Tensor,
    masks: np.ndarray,
    clean_masks: np.ndarray,
    min_num_frames: int,
) -> list[torch.Tensor]:
    """For each speaker, apply mask → interpolate → cyclic-tile to fbank_chunk.

    Returns one (T, 80) tensor per speaker slot; silent slots return the full chunk.
    """
    T = fbank_chunk.shape[0]
    result = []
    for spk in range(masks.shape[1]):
        mask = masks[:, spk]
        clean_mask = clean_masks[:, spk]
        used_mask = clean_mask if np.sum(clean_mask) > min_num_frames else mask
        mask_t = torch.from_numpy(used_mask).unsqueeze(0).unsqueeze(0).float()
        mask_interp = F.interpolate(mask_t, size=T, mode="nearest").squeeze().numpy()
        active = mask_interp > 0.5
        active_count = int(active.sum())
        if active_count == 0:
            result.append(fbank_chunk)
            continue
        active_fbank = fbank_chunk[active]
        indices = torch.arange(T) % active_count
        result.append(active_fbank[indices])
    return result


class AMISegmentationDataset(BaseDataset):
    """AMI Corpus IHM recordings chunked into 5-second waveform windows.

    Calibration input for PyannoteSegmentation.
    Each sample: waveform tensor of shape (1, 80000), float32, 16 kHz.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        dataset_path = _AMI_ASSETS[0].path.parent
        self._chunks: list[np.ndarray] = []
        super().__init__(dataset_path, split, input_spec)
        self._load_chunks()

    def _load_chunks(self) -> None:
        self._chunks = []
        for asset in _AMI_ASSETS:
            audio = _read_wav_mono_16k(asset.path)
            self._chunks.extend(_sliding_chunks(audio, _SEG_FRAMES, _SEG_STEP_FRAMES))

    def _download_data(self) -> None:
        for asset in _AMI_ASSETS:
            asset.fetch()

    def _validate_data(self) -> bool:
        return all(asset.path.exists() for asset in _AMI_ASSETS)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[Any]]:
        waveform = torch.from_numpy(self._chunks[index]).unsqueeze(0)
        return waveform, []

    @staticmethod
    def default_samples_per_job() -> int:
        return 32


class AMIEmbeddingDataset(BaseDataset):
    """AMI Corpus IHM recordings converted to log-mel fbank tensors.

    Calibration input for PyannoteEmbedding.
    Each sample: fbank tensor of shape (498, 80), float32.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        dataset_path = _AMI_ASSETS[0].path.parent
        self._fbanks: list[np.ndarray] = []
        super().__init__(dataset_path, split, input_spec)
        self._load_fbanks()

    def _load_fbanks(self) -> None:
        # Lazy import to avoid circular dependency (model.py imports this file)
        from qai_hub_models.models.pyannote_speaker_diarization.model import (
            load_pyannote_pipeline,
        )

        pipeline = load_pyannote_pipeline()
        _segmentation = pipeline._segmentation
        seg_model = _segmentation.model
        seg_model.cpu().eval()

        window_size = _SEG_FRAMES
        min_num_samples = pipeline._embedding.min_num_samples
        with torch.inference_mode():
            _dummy_out = seg_model(torch.zeros(1, 1, window_size))
        num_seg_frames = _dummy_out.shape[1]
        min_num_frames = math.ceil(num_seg_frames * min_num_samples / window_size)

        self._fbanks = []
        for asset in _AMI_ASSETS:
            audio = _read_wav_mono_16k(asset.path)
            chunks = _sliding_chunks(audio, _SEG_FRAMES, _SEG_STEP_FRAMES)
            # Stride to span the full meeting; targets ~100 chunks per recording.
            stride = max(1, len(chunks) // 100)
            for chunk in chunks[::stride]:
                waveform = torch.from_numpy(chunk).float().unsqueeze(0)

                with torch.inference_mode():
                    seg_out = seg_model(waveform.unsqueeze(0))
                    masks_raw = _segmentation.conversion(seg_out)
                masks = np.nan_to_num(
                    masks_raw.squeeze(0).cpu().numpy(), nan=0.0
                ).astype(np.float32)

                overlap = (masks.sum(axis=1, keepdims=True) >= 2).astype(np.float32)
                clean_masks = masks * (1.0 - overlap)

                fbank_chunk = pipeline._embedding.model_.compute_fbank(
                    waveform[None]
                ).squeeze(0)

                for fbank in _apply_speaker_masks(
                    fbank_chunk, masks, clean_masks, min_num_frames
                ):
                    self._fbanks.append(fbank.numpy())

    def _download_data(self) -> None:
        for asset in _AMI_ASSETS:
            asset.fetch()

    def _validate_data(self) -> bool:
        return all(asset.path.exists() for asset in _AMI_ASSETS)

    def __len__(self) -> int:
        return len(self._fbanks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[Any]]:
        fbank = torch.from_numpy(self._fbanks[index])
        return fbank, []

    @staticmethod
    def default_samples_per_job() -> int:
        return 32

    @staticmethod
    def default_num_calibration_samples() -> int:
        return 500


def _encode_path(path: str) -> torch.Tensor:
    """Encode a file path as a fixed-length uint8 tensor (padded / truncated to _PATH_TENSOR_LEN)."""
    b = path.encode()[:_PATH_TENSOR_LEN]
    b = b.ljust(_PATH_TENSOR_LEN, b"\x00")
    return torch.frombuffer(bytearray(b), dtype=torch.uint8)


def decode_path(tensor: torch.Tensor) -> str:
    """Decode a uint8 path tensor produced by _encode_path."""
    return bytes(tensor.numpy()).rstrip(b"\x00").decode()


class AMIDiarizationEvalDataset(BaseDataset):
    """AMI Corpus test.mini split for end-to-end DER evaluation.

    Each sample: (audio_path_tensor, rttm_path_tensor), both uint8 of length 512.
    Source: AMI IHM audio (CC BY 4.0) + pyannote/AMI-diarization-setup RTTM (Apache-2.0).
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        input_spec: InputSpec | None = None,
    ) -> None:
        dataset_path = _AMI_ASSETS[0].path.parent
        super().__init__(dataset_path, split, input_spec)

    def _download_data(self) -> None:
        for asset in _AMI_ASSETS:
            asset.fetch()
        for asset in _AMI_RTTM_ASSETS:
            asset.fetch()

    def _validate_data(self) -> bool:
        return all(asset.path.exists() for asset in _AMI_ASSETS) and all(
            asset.path.exists() for asset in _AMI_RTTM_ASSETS
        )

    def __len__(self) -> int:
        return len(_AMI_ASSETS)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_tensor = _encode_path(str(_AMI_ASSETS[index].path))
        rttm_tensor = _encode_path(str(_AMI_RTTM_ASSETS[index].path))
        return audio_tensor, rttm_tensor

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://groups.inf.ed.ac.uk/ami/corpus/",
            split_description="AMI IHM test.mini split (3 recordings)",
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return 1
