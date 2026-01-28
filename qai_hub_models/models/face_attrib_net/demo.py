# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json
import os
from pathlib import Path

from qai_hub_models.models.face_attrib_net.app import FaceAttribNetApp
from qai_hub_models.models.face_attrib_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceAttribNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "img_sample.png"
)


# Run FaceAttribNet end-to-end on a sample image.
def face_attrib_net_demo(
    app_cls: type[FaceAttribNetApp],
    model_type: type[FaceAttribNet],
    model_id: str,
    is_test: bool = False,
):
    """
    Runs a demo using the FaceAttribNet model and application class.

    Parameters
    ----------
    app_cls
        The application class to be instantiated.

    model_type
        The model class to be instantiated.

    model_id
        Model name string.

    is_test
        Indicates whether the demo is being run in a test context (e.g., from test.py).
        Defaults to False.

    """
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)
    # Load image
    orig_image = load_image(args.image)
    print("Model loaded")

    input_spec = FaceAttribNet.get_input_spec()["image"][0]
    model_input_shape = input_spec[-2], input_spec[-1]
    app = app_cls(model, model_input_shape)  # type: ignore[arg-type]
    output = app.run_inference_on_image(orig_image)

    if not is_test:
        assert isinstance(output, dict)
        output_path = (args.output_dir or str(Path() / "build")) + "/output.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(output, wf, ensure_ascii=False, indent=4)
        print(f"Model outputs are saved at: {output_path}")


def main(is_test: bool = False):
    """
    Parameters
    ----------
    is_test
        see `face_attrib_net_demo` for details.
    """
    face_attrib_net_demo(FaceAttribNetApp, FaceAttribNet, MODEL_ID, is_test)


if __name__ == "__main__":
    main()
