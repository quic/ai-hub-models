# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face_quantized.model import (
    MediaPipeFace,
    MediaPipeFaceQuantizable,
)
from qai_hub_models.utils.asset_loaders import load_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=8, help="Number of images to use."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where encodings should be stored. Defaults to ./build.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to folder containing face images to calibrate the model.",
    )

    args = parser.parse_args()
    image_files = os.listdir(args.dataset_dir)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(image_files)

    image_files = [Path(args.dataset_dir) / fp for fp in image_files[: args.num_iter]]

    model = MediaPipeFaceQuantizable.from_pretrained(
        face_detector_encodings=None, landmark_detector_encodings=None
    )

    def _calibrate_model(model: torch.nn.Module, args):
        app = MediaPipeFaceApp(MediaPipeFace.from_pretrained())
        model_is_landmark_detector, image_paths = args
        if model_is_landmark_detector:
            app.landmark_detector = model
        else:
            app.detector = model
        for image_path in tqdm(image_paths):
            image = load_image(image_path).convert("RGB")
            app.predict_landmarks_from_image(image)

    model.face_detector.quant_sim.compute_encodings(
        _calibrate_model, [False, image_files]
    )
    model.face_landmark_detector.quant_sim.compute_encodings(
        _calibrate_model, [True, image_files]
    )

    output_path = args.output_dir or str(Path() / "build")

    model.face_detector.quant_sim.save_encodings_to_json(
        output_path, "face_detector_quantized_encodings"
    )
    print(f"Wrote {output_path}/face_detector_quantized_encodings.json\n")

    model.face_landmark_detector.quant_sim.save_encodings_to_json(
        output_path, "landmark_detector_quantized_encodings"
    )
    print(f"Wrote {output_path}/landmark_detector_quantized_encodings.json\n")
