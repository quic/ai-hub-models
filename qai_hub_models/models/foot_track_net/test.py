# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pickle as pkl

import numpy as np
import pytest

from qai_hub_models.models.foot_track_net.app import FootTrackNet_App
from qai_hub_models.models.foot_track_net.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.foot_track_net.demo import main as demo_main
from qai_hub_models.models.foot_track_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FootTrackNet_model,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_path,
)
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_RST_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "oracle_rst1.pkl"
)


# Verify that the output from Torch is as expected. bbox, landmark, visibility
@skip_clone_repo_check
def test_task():
    app = FootTrackNet_App(FootTrackNet_model.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    objs_face, objs_person = app.det_image(original_image)

    pth = load_path(OUTPUT_RST_ADDRESS, "tmp")
    print("pth is", pth)
    with open(pth, "rb") as handle:
        objs_face_oracle, objs_person_oracle = pkl.load(handle)

    # extrac the oracle result
    faces_bbox_ora = np.array(
        [
            objs_face_oracle[0].box,
            objs_face_oracle[1].box,
        ]
    )

    persons_bbox_ora = np.array(
        [
            objs_person_oracle[0].box,
            objs_person_oracle[1].box,
        ]
    )

    persons_landmark_ora = np.array(
        [
            [
                objs_person_oracle[0].landmark[0],
                objs_person_oracle[0].landmark[15],
                objs_person_oracle[0].landmark[16],
            ],
            [
                objs_person_oracle[1].landmark[0],
                objs_person_oracle[1].landmark[15],
                objs_person_oracle[1].landmark[16],
            ],
        ]
    )

    persons_visibility_ora = np.array(
        [
            [objs_person_oracle[0].vis[15], objs_person_oracle[0].vis[16]],
            [objs_person_oracle[1].vis[15], objs_person_oracle[1].vis[16]],
        ]
    )

    # extract the key detection result
    faces_bbox = np.array(
        [
            objs_face[0].box,
            objs_face[1].box,
        ]
    )

    persons_bbox = np.array(
        [
            objs_person[0].box,
            objs_person[1].box,
        ]
    )

    persons_landmark = np.array(
        [
            [
                objs_person[0].landmark[0],
                objs_person[0].landmark[15],
                objs_person[0].landmark[16],
            ],
            [
                objs_person[1].landmark[0],
                objs_person[1].landmark[15],
                objs_person[1].landmark[16],
            ],
        ]
    )

    persons_visibility = np.array(
        [
            [objs_person[0].vis[15], objs_person[0].vis[16]],
            [objs_person[1].vis[15], objs_person[1].vis[16]],
        ]
    )

    # assert, face_bbox, person_bbox, person_landmark and person_landmark_visibility
    assert_most_close(
        np.asarray(faces_bbox_ora),
        np.asarray(faces_bbox),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_bbox_ora),
        np.asarray(persons_bbox),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_landmark_ora),
        np.asarray(persons_landmark),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_visibility_ora),
        np.asarray(persons_visibility),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    app = FootTrackNet_App(
        FootTrackNet_model.from_pretrained().convert_to_torchscript()
    )
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    objs_face, objs_person = app.det_image(original_image)

    pth = load_path(OUTPUT_RST_ADDRESS, "tmp")
    with open(pth, "rb") as handle:
        objs_face_oracle, objs_person_oracle = pkl.load(handle)

    # extrac the oracle result
    faces_bbox_ora = np.array(
        [
            objs_face_oracle[0].box,
            objs_face_oracle[1].box,
        ]
    )

    persons_bbox_ora = np.array(
        [
            objs_person_oracle[0].box,
            objs_person_oracle[1].box,
        ]
    )

    persons_landmark_ora = np.array(
        [
            [
                objs_person_oracle[0].landmark[0],
                objs_person_oracle[0].landmark[15],
                objs_person_oracle[0].landmark[16],
            ],
            [
                objs_person_oracle[1].landmark[0],
                objs_person_oracle[1].landmark[15],
                objs_person_oracle[1].landmark[16],
            ],
        ]
    )

    persons_visibility_ora = np.array(
        [
            [objs_person_oracle[0].vis[15], objs_person_oracle[0].vis[16]],
            [objs_person_oracle[1].vis[15], objs_person_oracle[1].vis[16]],
        ]
    )

    # extract the key detection result
    faces_bbox = np.array(
        [
            objs_face[0].box,
            objs_face[1].box,
        ]
    )

    persons_bbox = np.array(
        [
            objs_person[0].box,
            objs_person[1].box,
        ]
    )

    persons_landmark = np.array(
        [
            [
                objs_person[0].landmark[0],
                objs_person[0].landmark[15],
                objs_person[0].landmark[16],
            ],
            [
                objs_person[1].landmark[0],
                objs_person[1].landmark[15],
                objs_person[1].landmark[16],
            ],
        ]
    )

    persons_visibility = np.array(
        [
            [objs_person[0].vis[15], objs_person[0].vis[16]],
            [objs_person[1].vis[15], objs_person[1].vis[16]],
        ]
    )

    # assert, face_bbox, person_bbox, person_landmark and person_landmark_visibility
    assert_most_close(
        np.asarray(faces_bbox_ora),
        np.asarray(faces_bbox),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_bbox_ora),
        np.asarray(persons_bbox),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_landmark_ora),
        np.asarray(persons_landmark),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )
    assert_most_close(
        np.asarray(persons_visibility_ora),
        np.asarray(persons_visibility),
        diff_tol=0.01,
        atol=0.001,
        rtol=0.001,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
