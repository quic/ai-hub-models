// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

public class UtilsESRGAN extends Utils{

    public void PreProcess(Bitmap inputBitmap, int input_dims1, int input_dims2, float[][][][] floatinputarray){
        for (int x = 0; x < input_dims1; x++) {
            for (int y = 0; y < input_dims2; y++) {
                int pixel = inputBitmap.getPixel(x, y);
                floatinputarray[0][x][y][0] = Color.red(pixel);
                floatinputarray[0][x][y][1] = Color.green(pixel);
                floatinputarray[0][x][y][2] = Color.blue(pixel);
            }
        }
    }

    public void PostProcess(Bitmap outbmp, int output_dims1, int output_dims2, float[][][][] floatoutputarray) {
        for (int x = 0; x < output_dims1; x++) {
            for (int y = 0; y < output_dims2; y++) {
                int red = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][0])));
                int green = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][1])));
                int blue = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][2])));
                int color = Color.argb(255, red, green, blue);
                outbmp.setPixel(x, y, color);
            }
        }
    }
}
