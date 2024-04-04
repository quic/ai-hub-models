package com.qcom.aistack_objdetect;

import android.app.Application;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
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
    public native int inferQNN(long inputmataddress, int width,int height, float[][]boxcoords, String[] classname);


    /**
     * This method loads ML models on selected runtime
     */
    public boolean loadingMODELS(String runtime_var, String dlc_name) {
        assetManager = mApplication.getAssets();
        String nativeDirPath = mApplication.getApplicationInfo().nativeLibraryDir;
        String res_query = queryRuntimes(nativeDirPath);
        System.out.println(res_query);
        String[] path = {"./"};
//        assetManager.list(path);
//        System.out.println("list--sumith:"+);


        String init_str = initQNN(assetManager, runtime_var, dlc_name,nativeDirPath);
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
    public int qnnInference(Bitmap modelInputBitmap, int fps, ArrayList<RectangleBox> BBlist) {

        int result=0;

        try{

            Mat inputMat = new Mat();
            Utils.bitmapToMat(modelInputBitmap, inputMat);

            float[][] boxCoords = new float[100][5];  //Stores box coords for all person, MAXLIMIT is 100, last coords i.e. boxCoords[k][4] stores confidence value <-- IMP
            String[] boxnames = new String[100];

            System.out.println("Before inferQNN call ");
            int numhuman = inferQNN(inputMat.getNativeObjAddr(), modelInputBitmap.getWidth(), modelInputBitmap.getHeight(), boxCoords,boxnames);
            System.out.println("After inferQNN call ");
//            Log.i("QNNHelper", "numhuman: " + numhuman);
            System.out.println("numhuman: " + numhuman);

            if (numhuman == -1)
            {
                Log.e("QNNHelper", "Error loading model properly. Return error..");
                return -1;
            }

            for(int k=0;k<numhuman;k++) {
                System.out.println("Entered the BB loop");
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