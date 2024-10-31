# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable

import cv2
import numpy as np

ESCAPE_KEY_ID = 27


def capture_and_display_processed_frames(
    frame_processor: Callable[[np.ndarray], np.ndarray],
    window_display_name: str,
    cap_device: int = 0,
) -> None:
    """
    Capture frames from the given input camera device, run them through
    the frame processor, and display the outputs in a window with the given name.

    User should press Esc to exit.

    Inputs:
        frame_processor: Callable[[np.ndarray], np.ndarray]
            Processes frames.
            Input and output are numpy arrays of shape (H W C) with BGR channel layout and dtype uint8 / byte.
        window_display_name: str
            Name of the window used to display frames.
        cap_device: int
            Identifier for the camera to use to capture frames.
    """
    cv2.namedWindow(window_display_name)
    capture = cv2.VideoCapture(cap_device)
    if not capture.isOpened():
        raise ValueError("Unable to open video capture.")

    frame_count = 0
    has_frame, frame = capture.read()
    while has_frame:
        assert isinstance(frame, np.ndarray)

        frame_count = frame_count + 1
        # mirror frame
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

        # process & show frame
        processed_frame = frame_processor(frame)
        cv2.imshow(window_display_name, processed_frame[:, :, ::-1])

        has_frame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == ESCAPE_KEY_ID:
            break

    capture.release()
