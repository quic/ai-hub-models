// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.aihub_segmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import androidx.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;
import java.util.concurrent.locks.ReentrantLock;


/**
 * FragmentRender class is utility for segmentation on camera frames.
 * FragmentRender has utility in fragment_camera.xml and CameraFragment Class
 */

public class FragmentRender extends View {
    private final ReentrantLock mLock = new ReentrantLock();
    private Bitmap mBitmap = null;

    int[] ColorArray = {
            Color.GREEN,
            Color.RED, //aeroplane
            Color.RED, //bicycle
            Color.GREEN, //bird
            Color.RED, //boat
            Color.MAGENTA,  // 5 - bottle
            Color.RED, //bus
            Color.RED,  // 7 - car
            Color.GREEN, // cat
            Color.CYAN, // 9 - chair
            Color.GREEN, //cow
            Color.WHITE, //diningtable
            Color.GREEN, // dog
            Color.GREEN, //horse
            Color.RED, //motorbike
            Color.GREEN, // 15 - Person
            Color.WHITE, //pottedplant
            Color.GREEN, //sheep
            Color.BLUE, // sofa
            Color.RED, // 19 - train
            Color.BLUE // 20 - tv
    };

    int fps;
    float inferTime = 500.0f;
    float[] segmap = null;
    int width;
    int height;
    float renderTime;

    private final Paint mTextColor= new Paint();

    private final Paint mBorderColor= new Paint();

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        segmap = null;
        mBorderColor.setColor(Color.MAGENTA);
        mBorderColor.setStrokeWidth(6);
        mBorderColor.setAlpha(5);

        mTextColor.setColor(Color.RED);
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);

    }

    public void set_map(int[] map, int width, int height,int fps, float inferTime)
    {
        long rStart=System.nanoTime();
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.inferTime = inferTime;

        mBitmap = Bitmap.createBitmap(map, width, height, Bitmap.Config.ARGB_8888);
        postInvalidate();
        renderTime=(System.nanoTime()-rStart)/1000000.0f;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        mLock.lock();

        if(width!=0) {
            long rStart = System.nanoTime();
            canvas.drawBitmap(mBitmap, 0,0,null);
            renderTime = (System.nanoTime() - rStart) / 1000000.0f;
            canvas.drawText("Infer: " + String.format("%.2f", inferTime) + "ms", 10, 70, mTextColor);
            canvas.drawText("FPS: " + fps, 10, 120, mTextColor);
            canvas.drawText("Render: " + String.format("%.2f", renderTime) + "ms", 10, 170, mTextColor);
        }
        mLock.unlock();
    }
}
