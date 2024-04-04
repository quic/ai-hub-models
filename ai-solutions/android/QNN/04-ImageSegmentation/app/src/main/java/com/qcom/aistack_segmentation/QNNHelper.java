package com.qcom.aistack_segmentation;

import android.app.Application;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import java.util.ArrayList;

public class QNNHelper {
    private final Application mApplication;
    private AssetManager assetManager;

    private float infer_time=0;

    // Constructor
    public QNNHelper(Application application) {
        mApplication = application;
    }
    public float getInfer_time()
    {return infer_time;}

    //Native functions
    public native String queryRuntimes(String backend);
    public native String initQNN(AssetManager assetManager, String backend, String model_name,String nativeDirPath);
    public native float inferQNN(long inputmataddress, int width,int height, float[] segment_pixel);


    /**
     * This method loads ML models on selected runtime
     */
    public boolean loadingMODELS(String runtime_var, String model_name) {
        assetManager = mApplication.getAssets();
        String nativeDirPath = mApplication.getApplicationInfo().nativeLibraryDir;
        String res_query = queryRuntimes(nativeDirPath);
        System.out.println(res_query);
        String[] path = {"./"};
//        assetManager.list(path);
//        System.out.println("list--sumith:"+);


        String init_str = initQNN(assetManager, runtime_var, model_name,nativeDirPath);
        System.out.println("RESULT:"+init_str);

        int success_count = init_str.split("success", -1).length -1;

        if(success_count>=1)
        {
            System.out.println("Model built successfully");
            return true;
        }

        return false;
    }


    /*
        This method makes inference on bitmap.
    */
    public float qnnInference(Bitmap modelInputBitmap, float[] segmap) {

        try{
            Mat inputMat = new Mat();
            Utils.bitmapToMat(modelInputBitmap, inputMat);


            Trace.beginSection("nativeTime");
            float inferTime = inferQNN(inputMat.getNativeObjAddr(), modelInputBitmap.getWidth(), modelInputBitmap.getHeight(), segmap);
            Trace.endSection();
            System.out.println("After returning to qnnInference, inferTime = " + inferTime);

            return inferTime;

        }catch (Exception e) {
            e.printStackTrace();
        }
        return -1.0f;
    }
}