# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image

from qai_hub_models.utils.display import is_headless, is_running_in_notebook


## Helper routines
def show_image(
    image: Image, masks: np.ndarray | None = None, output_dir: str | None = None
) -> None:
    """Show input image with mask applied"""
    if masks is None:
        print("No masks detected, so no images to display.")
        return

    should_display_in_notebook = is_running_in_notebook()
    should_save_to_disk = not should_display_in_notebook and (
        is_headless() or output_dir is not None
    )

    if should_display_in_notebook:
        print(
            "In python notebooks, make sure inline plotting is enabled via running '%matplotlib inline'"
        )

    if should_save_to_disk:
        if output_dir is None:
            raise ValueError(
                "In headless mode, --output-dir must be passed to choose where outputs are saved to disk."
            )
        os.makedirs(output_dir, exist_ok=True)
        print("Saving images to disk: ")

    for mask_idx, mask in enumerate(np.split(masks, masks.shape[1], 1)):
        plt.figure(num=mask_idx, figsize=(10, 10))
        plt.title(f"Predicted Mask {mask_idx}")
        plt.imshow(image)
        _draw_mask(mask)
        plt.axis("off")
        if should_save_to_disk:
            assert output_dir is not None
            outpath = os.path.join(output_dir, f"demo_output_mask{mask_idx}.jpg")
            plt.savefig(
                outpath,
                bbox_inches="tight",
            )
            print(f"   {outpath}")

    if not should_save_to_disk:
        plt.show()


def _draw_mask(mask: np.ndarray) -> None:
    """Helper routine to add mask over existing plot"""
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
