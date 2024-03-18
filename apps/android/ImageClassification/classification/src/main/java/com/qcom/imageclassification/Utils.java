// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imageclassification;


import android.graphics.Bitmap;
import android.graphics.Color;
import java.util.Arrays;
import java.util.List;

public class Utils {

    //PRE PROCESSING Model Input
    public void PreProcess(Bitmap inputBitmap, int input_dims1, int input_dims2, float[][][][] floatinputarray, List<Float> ImageMean, List<Float> ImageStd){
        for (int x = 0; x < input_dims1; x++) {
            for (int y = 0; y < input_dims2; y++) {
                int pixel = inputBitmap.getPixel(x, y);
                List<Float> rgb = Arrays.asList((float)Color.red(pixel), (float)Color.green(pixel), (float)Color.blue(pixel));
                for(int z = 0;z<3; z++){
                    floatinputarray[0][x][y][z] = (float)((rgb.get(z))-ImageMean.get(z))/ImageStd.get(z);
                }
            }
        }
    }
}
