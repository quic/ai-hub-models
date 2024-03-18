// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imageclassification;

import android.graphics.Bitmap;
import android.media.Image;

import java.util.ArrayList;
import java.util.List;

public class ImageClassificationResult {

    private List<Integer> topindices;
    private String ResultString;

    public ImageClassificationResult(List<Integer> customlist, String res)
    {
        this.topindices = customlist;
        this.ResultString = res;
    }

    public List<Integer> getIndices()
    {
        return  topindices;
    }
    public String getResultString()
    {
        return ResultString;
    }
}
