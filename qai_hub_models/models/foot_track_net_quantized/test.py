# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json

import numpy as np
import torch

from qai_hub_models.models._shared.foot_track_net.app import FootTrackNet_App
from qai_hub_models.models.foot_track_net.demo import main as demo_main
from qai_hub_models.models.foot_track_net.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.foot_track_net_quantized.model import FootTrackNetQuantizable
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_path,
)
from qai_hub_models.utils.image_processing import resize_pad
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test6.jpg"
)  # from float version
OUTPUT_RST_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test6_rst.json"
)


# Verify that the output from Torch is as expected. bbox, landmark, visibility
@skip_clone_repo_check
def test_task():
    # error tolerance
    diff_tol = 0.1
    atol = 5
    rtol = 0.1

    app = FootTrackNet_App(FootTrackNetQuantizable.from_pretrained())

    (_, _, height, width) = FootTrackNetQuantizable.get_input_spec()["image"][0]
    orig_image = np.array(load_image(INPUT_IMAGE_ADDRESS))
    orig_image = torch.from_numpy(orig_image).permute(2, 0, 1).unsqueeze_(0)
    image, scale, padding = resize_pad(orig_image, (height, width))
    objs_face, objs_person = app.det_image(image)
    pth = load_path(OUTPUT_RST_ADDRESS, "tmp")
    with open(pth) as f:
        rst = json.load(f)

    # extrac the oracle result
    faces_bbox_ora = np.array(rst["faces_bbox"])
    persons_bbox_ora = np.array(rst["persons_bbox"])
    persons_landmark_ora = np.array(rst["persons_landmark"])
    persons_visibility_ora = np.array(rst["persons_visibility"])

    # extract the key detection result
    faces_bbox = np.array([objs_face[0].box])
    persons_bbox = np.array([objs_person[0].box])
    assert objs_person[0].landmark is not None
    persons_landmark = np.array(
        [
            [
                objs_person[0].landmark[0],
                objs_person[0].landmark[15],
                objs_person[0].landmark[16],
            ]
        ]
    )

    assert objs_person[0].vis is not None
    persons_visibility = np.array([[objs_person[0].vis[15], objs_person[0].vis[16]]])

    # assert, face_bbox, person_bbox, person_landmark and person_landmark_visibility
    assert_most_close(
        np.asarray(faces_bbox_ora),
        np.asarray(faces_bbox),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.asarray(persons_bbox_ora),
        np.asarray(persons_bbox),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.asarray(persons_landmark_ora),
        np.asarray(persons_landmark),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.asarray(persons_visibility_ora),
        np.asarray(persons_visibility),
        diff_tol=diff_tol,
        atol=0.2,
        rtol=rtol,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
