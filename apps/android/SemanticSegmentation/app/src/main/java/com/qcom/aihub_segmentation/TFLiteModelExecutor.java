// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.aihub_segmentation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import com.qualcomm.qti.QnnDelegate;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

public class TFLiteModelExecutor {

    private Interpreter tfLite = null;
    private Interpreter tfLiteQNNDelegate = null;
    private int inpDims_h,inpDims_w;
    private int outDims_h,outDims_w;
    public long inferTime;
    public boolean mNetworkLoadedTFLITE = false;
    public boolean mNetworkLoadedQNNDelegate = false;
    private static final String TAG = "Segmentation_Inference";
    private ReentrantLock mLock = new ReentrantLock();


    public void initializingModel(Context context, String TFLITE_FILE) throws IOException {

        MappedByteBuffer tfLiteModel;

        try {
            tfLiteModel = loadModelFile(context.getApplicationContext().getAssets(), TFLITE_FILE);
        } catch (Exception e) {
            Log.e(TAG,"TFLite Model Loading Unsuccessful");
            throw new RuntimeException(e);
        }

        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLiteOptions.setNumThreads(4);
        tfLiteOptions.setUseXNNPACK(true);
        tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
        inpDims_h = tfLite.getInputTensor(0).shape()[1]; // B x H x W x C
        inpDims_w = tfLite.getInputTensor(0).shape()[2];
        outDims_h = tfLite.getOutputTensor(0).shape()[1]; // B x H x W x C
        outDims_w = tfLite.getOutputTensor(0).shape()[2];

        mNetworkLoadedTFLITE = true;
        Log.d(TAG, "TFLITE MODEL LOADED SUCCESSFULLY");

        QnnDelegate.Options options = new QnnDelegate.Options();
        options.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
        options.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST);
        options.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);
        options.setSkelLibraryDir(context.getApplicationInfo().nativeLibraryDir);
        tfLiteOptions.addDelegate(new QnnDelegate(options));
        tfLiteQNNDelegate = new Interpreter(tfLiteModel, tfLiteOptions);
        mNetworkLoadedQNNDelegate = true;
        Log.d(TAG, "QNNDELEGATE MODEL LOADED SUCCESSFULLY");

    }

    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws Exception {
        try{
            AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
        catch (Exception e)
        {
            throw e;
        }
    }
    public int[] tfIteInference(Bitmap modelInputBitmap, char runtime_var) {
        mLock.lock();

        long preprocessing_StartTime = System.nanoTime();

        int[] segMap = new int[modelInputBitmap.getWidth()*modelInputBitmap.getHeight()];
        try{

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(modelInputBitmap,inpDims_w, inpDims_h,true);
            float[][][][] floatInputArray = new float[1][inpDims_h][inpDims_w][3];
            for (int x = 0; x < inpDims_h; x++) {
                for (int y = 0; y < inpDims_w; y++) {
                    int pixel = scaledBitmap.getPixel(y, x);
                    floatInputArray[0][x][y][0] = (((float)Color.red(pixel))/225f);
                    floatInputArray[0][x][y][1] = ((float)Color.green(pixel)/255f);
                    floatInputArray[0][x][y][2] = ((float)Color.blue(pixel)/255f);
                }
            }

            Object[] inputArray = {floatInputArray};

            int noOfClasses = tfLite.getOutputTensor(0).shape()[3];
            float[][][][] output  = new float[1][outDims_h][outDims_w][noOfClasses];

            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0,output);

            long inferenceStartTime = System.nanoTime();
            long preprocTimeDiff=inferenceStartTime-preprocessing_StartTime;

            Log.d(TAG,"Preprocessing Time: "+preprocTimeDiff/1000000+" ms");

            if( runtime_var == 'N')
                tfLiteQNNDelegate.runForMultipleInputsOutputs(inputArray, outputMap);
            else
                tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

            long inference_EndTime = System.nanoTime();
            inferTime=(inference_EndTime-inferenceStartTime)/1000000;
            Log.d(TAG,"Inference Time: "+inferTime+" ms");

            Mat src = new Mat(outDims_h,outDims_w, CvType.CV_32FC1);

            float[] temparr = new float[outDims_w];
            for (int i = 0; i < outDims_h; i++) {

                for (int j = 0; j < outDims_w; j++) { // Looping through the columns
                    float max_val=Float.NEGATIVE_INFINITY;
                    float maxIndex =-1;  // Initializing the max value
                    for (int k = 0; k < noOfClasses; k++) { // Looping through the remaining elements in the first axis
                        if (output[0][i][j][k] > max_val) { // Comparing the current element with the max value
                            maxIndex=k;
                            max_val = output[0][i][j][k]; // Updating the max value if the current element is large
                        }
                    }
                    temparr[j]=maxIndex;
                }
                src.put(i,0,temparr);
            }

            Mat dst = new Mat(modelInputBitmap.getHeight(),modelInputBitmap.getWidth(), CvType.CV_8UC3);

            // Resize the image
            Imgproc.resize(src, dst, dst.size(), 0, 0, Imgproc.INTER_AREA);
            int z=0;
            for (int x = 0; x <modelInputBitmap.getHeight(); x++) {
                for (int y = 0; y <modelInputBitmap.getWidth(); y++){
                    segMap[z++] =(int)dst.get(x, y)[0];
                }
            }

            mLock.unlock();
            return segMap;

        }catch (Exception e) {
            e.printStackTrace();
        }
        return new int[modelInputBitmap.getWidth()*modelInputBitmap.getHeight()];
    }
}
