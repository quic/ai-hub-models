// -*- mode: java -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
package com.qcom.aistack_segmentation;

import android.app.Application;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class SNPEHelper {
    private final Application mApplication;
    private AssetManager assetManager;

    // Constructor
    public SNPEHelper(Application application) {
        mApplication = application;
    }

    //Native functions
    public native String queryRuntimes(String a);
    public native String initSNPE(AssetManager assetManager, char a, String dlc_name);
    public native float inferSNPE(long inputmataddress, int width,int height, float[] segment_pixel);

    /**
     * This method loads ML models on selected runtime
     */
    public boolean loadingMODELS(char runtime_var, String dlc_name) {

        assetManager = mApplication.getAssets();
        String nativeDirPath = mApplication.getApplicationInfo().nativeLibraryDir;
        String res_query = queryRuntimes(nativeDirPath);
        System.out.println(res_query);
        String tt = initSNPE(assetManager, runtime_var, dlc_name);
        System.out.println("RESULT:"+tt);

        int success_count = tt.split("success", -1).length -1;

        if(success_count==1)
        {
            System.out.println("Model built successfully");
            return true;
        }

        return false;
    }

    /*
        This method makes inference on bitmap.
    */
    public float snpeInference(Bitmap modelInputBitmap, float[] segmap) {

        try{

            Mat inputMat = new Mat();
            Utils.bitmapToMat(modelInputBitmap, inputMat);


            Trace.beginSection("nativeTime");
            float inferTime = inferSNPE(inputMat.getNativeObjAddr(), modelInputBitmap.getWidth(), modelInputBitmap.getHeight(), segmap);
            Trace.endSection();


            return inferTime;

        }catch (Exception e) {
                e.printStackTrace();
        }
        return -1.0f;
    }

}