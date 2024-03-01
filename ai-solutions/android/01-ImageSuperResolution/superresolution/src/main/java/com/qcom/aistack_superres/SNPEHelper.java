// -*- mode: java -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
package com.qcom.aistack_superres;

import static android.graphics.Color.rgb;

import android.app.Application;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

public class SNPEHelper {
    private final Application mApplication;
    private AssetManager assetManager;

    private float infer_time=0;

    // Constructor
    public SNPEHelper(Application application) {
        mApplication = application;
    }
    public float getInfer_time()
        {return infer_time;}

    //Native functions
    public native String queryRuntimes(String a);
    public native String initSNPE(AssetManager assetManager, char a, String dlc_name);
    public native float inferSNPE(long inputmataddress, long outputmataddress);

    /**
     * This method loads ML models on selected runtime
     */
    public boolean loadingMODELS(char runtime_var, String dlc_name) {

        assetManager = mApplication.getAssets();
        String nativeDirPath = mApplication.getApplicationInfo().nativeLibraryDir;
        String res_query = queryRuntimes(nativeDirPath);
        System.out.println(res_query);
        String init_str = initSNPE(assetManager, runtime_var, dlc_name);
        System.out.println("RESULT:"+init_str);

        int success_count = init_str.split("success", -1).length -1;

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
    public Bitmap snpeInference(Bitmap modelInputBitmap) {

        try{

            Mat inputMat = new Mat();
            Utils.bitmapToMat(modelInputBitmap, inputMat);

            Mat outputMat = new Mat();

            infer_time = inferSNPE(inputMat.getNativeObjAddr(), outputMat.getNativeObjAddr());


            if(infer_time==0.0)
                System.out.println("ERROR");
            else
            {
                Bitmap outputBitmap =  Bitmap.createBitmap(outputMat.cols(), outputMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(outputMat,outputBitmap);
                return outputBitmap;
            }
        }catch (Exception e) {
                e.printStackTrace();
        }
        return null;
    }


}