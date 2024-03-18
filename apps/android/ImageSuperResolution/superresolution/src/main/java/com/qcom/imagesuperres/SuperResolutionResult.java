// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;

import android.graphics.Bitmap;

public class SuperResolutionResult {
    private final Bitmap[] highResolutionImages;

    public SuperResolutionResult(Bitmap[] highResolutionImages) {
        this.highResolutionImages = highResolutionImages;
    }

    public Bitmap[] getHighResolutionImages() {
        return highResolutionImages;
    }
}
