# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np


## Helper routines
def show_image(image, masks=None):
    """Show input image with mask applied"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if masks is not None:
        _show_mask(masks, plt.gca())
    plt.axis("off")
    plt.show()


def _show_mask(mask, ax):
    """Helper routine to add mask over existing plot"""
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
