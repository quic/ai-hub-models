// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;


import android.content.Context;
import android.graphics.Bitmap;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.qualcomm.qti.QnnDelegate;
import org.tensorflow.lite.Interpreter;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import android.widget.ImageView;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;

public class SuperResolution {
    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private Interpreter tfLite_QNN;

    private boolean model_loaded= false;

    private QnnDelegate qnnDelegate = null;
    private static final String TAG = "SUPERRES";
    private static Utils util = new Utils();


    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void close()
    {
        if(qnnDelegate!=null) {
            qnnDelegate.close();
        }

        if(tfLite != null){
            tfLite.close();
        }

        if(tfLiteModel!=null)
            tfLiteModel.clear();
    }

    public boolean getBuildStatus()
    {
        return model_loaded;
    }
    public boolean initializeModel(Context context, String tflitemodelfileName) {

        //If modeltype is in Red, it will resolved after building the app
        String kk = context.getString(R.string.modeltype);
        Log.i(TAG,"MY STRING FROM GetProperty is : "+kk);

        if(kk == "ESRGAN")
            util = new UtilsESRGAN();

        try {
            tfLiteModel = loadModelFile(context.getApplicationContext().getAssets(), tflitemodelfileName);
            Log.i(TAG, "MODEL LOADED");
            Interpreter.Options tfLiteOptions = new Interpreter.Options();
            tfLiteOptions.setNumThreads(4);
            tfLiteOptions.setUseXNNPACK(true);
            tfLite = new Interpreter(tfLiteModel, tfLiteOptions);

            QnnDelegate.Options options = new QnnDelegate.Options();
            options.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
            options.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST);
            options.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);

            Log.i(TAG, "NATIVE LIB PATH: " + context.getApplicationInfo().nativeLibraryDir);
            options.setSkelLibraryDir(context.getApplicationInfo().nativeLibraryDir);
            qnnDelegate = new QnnDelegate(options);
            tfLiteOptions.addDelegate(qnnDelegate);
            tfLite_QNN = new Interpreter(tfLiteModel,tfLiteOptions);
            Log.i(TAG, "QnnDelegate Option Added ");
            model_loaded= true;
            return true;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return false;
        }

    }

    public Result<SuperResolutionResult> inference(Bitmap[] images, String backend) {
        System.out.println("Processing %d images %dx%d."+ images.length+ images[0].getWidth()+ images[0].getHeight());
        String remarks = "";
        try{
            int[] arr = tfLite.getInputTensor(0).shape();
            int input_dims1 = arr[1];
            int input_dims2 = arr[2];

            if(input_dims1!=input_dims2)
            {
                remarks = "THIS APP IS DESIGNED FOR 1:1 ASPECT RATIO";
            }
            //PREPROCESSING INPUT to Model input Shape and Normalizing data
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(images[0],input_dims1,input_dims2,true);
            float[][][][] floatinputarray = new float[1][input_dims1][input_dims2][3];
            util.PreProcess(scaledBitmap,input_dims1,input_dims2,floatinputarray);

            Object[] inputArray = {floatinputarray};
            int[] out_arr = tfLite.getOutputTensor(0).shape();
            int output_dims1 = out_arr[1];
            int output_dims2 = out_arr[2];

            float[][][][] floatoutputarray = new float[1][output_dims1][output_dims2][3];
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, floatoutputarray);

            Log.i(TAG, "inputTensor shape"+ Arrays.toString(tfLite.getInputTensor(0).shape()));
            long inferenceStartTime = System.nanoTime();
            if (backend.equals("QNNDELEGATE") && tfLite_QNN != null) {
                System.out.println("QNN BACKEND");
                tfLite_QNN.runForMultipleInputsOutputs(inputArray, outputMap);
            }
            else if (backend.equals("TFLITE") && tfLite != null) {
                System.out.println("TFLITE BACKEND");
                tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
            }
            else
            {
                System.out.println("PROBLEM WITH Model Iinitilization");
            }
            long inferenceEndTime = System.nanoTime();
            Log.i(TAG,"MODEL EXECUTED");
            System.out.println("Inference time: "+ (inferenceEndTime - inferenceStartTime) / 1000);// calculated inference time


            Bitmap outbmp = Bitmap.createBitmap(output_dims1, output_dims2, Bitmap.Config.ARGB_8888);
            util.PostProcess(outbmp, output_dims1, output_dims2, floatoutputarray);

            Bitmap[] finalProcessedImages = new Bitmap[images.length];
            finalProcessedImages[0] = outbmp;

            SuperResolutionResult result = new SuperResolutionResult(finalProcessedImages);
            return new Result<>(result,
                    (inferenceEndTime - inferenceStartTime) / 1000000, remarks);

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }
}
