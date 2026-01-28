# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import soundfile
import torch

from qai_hub_models.models.zipformer.app import (
    ZipformerApp,
    create_streaming_feature_extractor,
    streaming_greedy_search,
)
from qai_hub_models.models.zipformer.demo import load_demo_audio
from qai_hub_models.models.zipformer.demo import main as demo_main
from qai_hub_models.models.zipformer.model import HfZipformer


def test_demo() -> None:
    demo_main(is_test=True)


def test_transcribe() -> None:
    """
    Test that HfWhisperApp produces end to end transcription results that
    matches with the original model
    """
    model = HfZipformer.from_pretrained()
    app = ZipformerApp(model.encoder, model.decoder, model.joiner, model)
    audio, sample_rate = soundfile.read(load_demo_audio())
    transcription_app = app.transcribe(audio, sample_rate)

    online_fbank = create_streaming_feature_extractor()  # 100 frames / s
    online_fbank.accept_waveform(sampling_rate=sample_rate, waveform=audio)
    frames_ = np.array(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    frames = torch.tensor(frames_, dtype=torch.float32).unsqueeze(0)  # (1, len, 80)
    # tokens = self.transcribe_tokens(audio, audio_sample_rate)
    origin_fpm = {
        "encoder": model.encoder,
        "decoder": model.decoder,
        "joiner": model.joiner,
    }
    text = streaming_greedy_search(model, origin_fpm, frames)
    print(transcription_app, text)
    assert transcription_app == text
