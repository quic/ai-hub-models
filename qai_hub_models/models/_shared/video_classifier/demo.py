# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.video_classifier.app import KineticsClassifierApp
from qai_hub_models.models._shared.video_classifier.model import KineticsClassifier
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_path, qaihm_temp_dir


#
# Run KineticsClassifierApp end-to-end on a sample video.
# The demo will display top classification predictions for the video.
#
def kinetics_classifier_demo(
    model_type: type[KineticsClassifier],
    default_video: str | CachedWebAsset,
    is_test: bool = False,
    app_cls: type[KineticsClassifierApp] = KineticsClassifierApp,
):
    # Demo parameters
    parser = get_model_cli_parser(model_type)

    parser.add_argument(
        "--video", type=str, default=default_video, help="video file path or URL."
    )

    args = parser.parse_args([] if is_test else None)

    # Load image & model
    model = model_from_cli_args(model_type, args)
    app = app_cls(model)
    print("Model Loaded")
    with qaihm_temp_dir() as tmpdir:
        dst_path = load_path(args.video, tmpdir)
        predictions = app.predict(path=str(dst_path))
    top5_classes = ", ".join(predictions)
    if not is_test:
        print(f"Top 5 predictions: {top5_classes}")
