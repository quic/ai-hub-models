# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.extern.mmpose import patch_mmpose_no_build_deps

with patch_mmpose_no_build_deps():
    from mmpose.apis import MMPoseInferencer
    from mmpose.apis.inferencers import Pose3DInferencer
    from mmpose.apis.inferencers.base_mmpose_inferencer import BaseMMPoseInferencer


def _set_mmpose_base_inferencer_show_progress(
    inferencer: BaseMMPoseInferencer, show_progress: bool = False
) -> BaseMMPoseInferencer:
    inferencer.show_progress = show_progress
    if hasattr(inferencer, "detector") and inferencer.detector is not None:
        inferencer.detector.show_progress = show_progress
    if isinstance(inferencer, Pose3DInferencer):
        _set_mmpose_base_inferencer_show_progress(
            inferencer.pose2d_model, show_progress
        )
    return inferencer


def set_mmpose_inferencer_show_progress(
    inferencer: MMPoseInferencer, show_progress: bool = False
) -> MMPoseInferencer:
    """
    Disables verbose logging output when the inferencer object runs the model.
    Returns the input inferencer. The object is modified in-place.
    """
    _set_mmpose_base_inferencer_show_progress(inferencer, show_progress)
    _set_mmpose_base_inferencer_show_progress(inferencer.inferencer, show_progress)
    return inferencer


__all__ = ["set_mmpose_inferencer_show_progress"]
