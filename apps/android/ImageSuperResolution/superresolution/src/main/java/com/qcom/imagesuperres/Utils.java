// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;

import android.graphics.Bitmap;
import android.graphics.Color;

public class Utils {

    public void PreProcess(Bitmap inputBitmap, int input_dims1, int input_dims2, float[][][][] floatinputarray){
        for (int x = 0; x < input_dims1; x++) {
            for (int y = 0; y < input_dims2; y++) {
                int pixel = inputBitmap.getPixel(x, y);
                // Normalize channel values to [-1.0, 1.0]. Here, pixel values
                // are positive so the effective range will be [0.0, 1.0]
                floatinputarray[0][x][y][0] = (Color.red(pixel))/255.0f;
                floatinputarray[0][x][y][1] = (Color.green(pixel))/255.0f;
                floatinputarray[0][x][y][2] = (Color.blue(pixel))/255.0f;
            }
        }
    }

    public void PostProcess(Bitmap outbmp, int output_dims1, int output_dims2, float[][][][] floatoutputarray) {
        for (int x = 0; x < output_dims1; x++) {
            for (int y = 0; y < output_dims2; y++) {
                int red = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][0] * 255)));
                int green = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][1] * 255)));
                int blue = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][2] * 255)));
                int color = Color.argb(255, red, green, blue);
                outbmp.setPixel(x, y, color);
            }
        }
    }
}
