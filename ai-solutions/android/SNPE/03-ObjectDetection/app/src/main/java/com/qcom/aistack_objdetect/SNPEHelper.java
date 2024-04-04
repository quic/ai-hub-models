// -*- mode: java -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
package com.qcom.aistack_objdetect;

import android.app.Application;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import java.util.ArrayList;

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
    public native int inferSNPE(long inputmataddress, int width,int height, float[][]boxcoords, String[] classname);


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
    public int snpeInference(Bitmap modelInputBitmap, int fps, ArrayList<RectangleBox> BBlist) {

        int result=0;

        try{

            Mat inputMat = new Mat();
            Utils.bitmapToMat(modelInputBitmap, inputMat);

            float[][] boxCoords = new float[100][5];  //Stores box coords for all person, MAXLIMIT is 100, last coords i.e. boxCoords[k][4] stores confidence value <-- IMP
            String[] boxnames = new String[100];

            //System.out.println("call inferSNPE input_Width="+modelInputBitmap.getWidth()+" input_Height="+modelInputBitmap.getHeight());

            int numhuman = inferSNPE(inputMat.getNativeObjAddr(), modelInputBitmap.getWidth(), modelInputBitmap.getHeight(), boxCoords,boxnames);

            if (numhuman == -1)
            {
                Log.e("SNPEHelper", "Error loading model properly. Return error..");
                return -1;
            }

            for(int k=0;k<numhuman;k++) {
                RectangleBox tempbox = new RectangleBox();

                tempbox.top = boxCoords[k][0];
                tempbox.bottom = boxCoords[k][1];
                tempbox.left = boxCoords[k][2];
                tempbox.right = boxCoords[k][3];
                tempbox.fps = fps;
                tempbox.processing_time = String.valueOf(boxCoords[k][4]);
                tempbox.label = boxnames[k];

                BBlist.add(tempbox);
            }

        }catch (Exception e) {
                e.printStackTrace();
        }

        return result;
    }

}