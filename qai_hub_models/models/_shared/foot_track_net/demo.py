# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models._shared.foot_track_net.app import (
    BBox_landmarks,
    FootTrackNet_App,
    drawbbox,
)
from qai_hub_models.models.foot_track_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FootTrackNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import resize_pad

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test1.jpg"
)


def undo_resize_pad_BBox(bbox: BBox_landmarks, scale: float, padding: tuple[int, int]):
    """
    undo the resize and pad in place of the BBox_landmarks object.
    operation in place to replace the inner coordinates
    Parameters:
        scale: single scale from original to target image.
        pad: left, top padding size
    Return:
        None.
    """
    if bbox.landmark is not None:
        for lmk in bbox.landmark:
            lmk[0] = (lmk[0] - padding[0]) / scale
            lmk[1] = (lmk[1] - padding[1]) / scale
    bbox.x = (bbox.x - padding[0]) / scale
    bbox.y = (bbox.y - padding[1]) / scale
    bbox.r = (bbox.r - padding[0]) / scale
    bbox.b = (bbox.b - padding[1]) / scale


def demo(model_cls: type[FootTrackNet], is_test: bool = False):
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image
    (_, _, height, width) = model_cls.get_input_spec()["image"][0]

    orig_image = np.array(load_image(args.image))
    orig_image = orig_image.astype(np.float32)
    img = torch.from_numpy(orig_image).permute(2, 0, 1).unsqueeze_(0)
    image, scale, padding = resize_pad(img, (height, width))

    print("Model Loaded")
    # OnDeviceModel is underspecified for this usage
    app = FootTrackNet_App(model)  # pyright: ignore[reportArgumentType]
    objs_face, objs_person = app.det_image(image)
    objs = objs_face + objs_person

    img_out = np.array(orig_image)[:, :, ::-1].copy()  # to BGR
    jt_vis = [0, 15, 16]
    vis_thr = 0.5
    color_maps = create_color_map(2)

    for obj in objs:
        undo_resize_pad_BBox(obj, scale, padding)
        color = color_maps[int(obj.label)]
        color = [float(e) for e in color]
        vis = obj.vis
        img_out = drawbbox(
            img_out,
            obj,
            color=color,
            landmarkcolor=color,
            visibility=vis,
            joint_to_visualize=jt_vis,
            visibility_thresh=vis_thr,
        )
    img_out = img_out.astype(np.uint8)
    img_out_PIL = Image.fromarray(img_out[:, :, ::-1])

    if not is_test:
        display_or_save_image(
            img_out_PIL, args.output_dir, "FootTrackNet_demo_output.png"
        )
