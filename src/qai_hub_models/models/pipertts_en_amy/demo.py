# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.pipertts_en_amy.model import PiperTTS_EN_AMY
from qai_hub_models.models.templates.pipertts.app import DEFAULT_TEXTS, PiperTTSApp
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(PiperTTS_EN_AMY)

    args = parser.parse_args([] if is_test else None)
    model = model_from_cli_args(PiperTTS_EN_AMY, args)
    app = PiperTTSApp(
        model.encoder, model.sdp, model.flow, model.decoder, model.get_language()
    )
    audio_path = app.predict(DEFAULT_TEXTS[model.get_language()])
    print(f"Audio generated and saved to {audio_path}")


if __name__ == "__main__":
    main()
