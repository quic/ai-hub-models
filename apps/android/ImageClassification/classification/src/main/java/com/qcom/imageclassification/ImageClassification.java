// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
import android.widget.Toast;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;

public class ImageClassification {

    private Context context;
    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private Interpreter tfLite_QNN;
    private QnnDelegate qnnDelegate = null;
    private static final String TAG = "Sahin";
    private static final float IMAGE_MEAN = 127.7f;
    private static final float IMAGE_STD =128f;
    private List<String> labelList;
    private static final String LABEL_PATH = "labels.txt";
    boolean model_loaded = false;
    public boolean getBuildStatus()
    {
        return model_loaded;
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

        if(labelList!=null)
            labelList.clear();
    }
    public boolean initializeModel(Context context,String TFLITE_FILE) throws IOException {

        this.context = context;

        try {
            tfLiteModel = loadModelFile(context.getApplicationContext().getAssets(), TFLITE_FILE);
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
            Log.i(TAG, "QnnDelegate Option Added");
            model_loaded= true;
            Log.d(TAG,"Label list Loaded Successfully");
            labelList =loadLabelList(LABEL_PATH);
            return true;

        } catch (IOException e) {
            Log.e(TAG,"TFLite Model Loading Unsuccessfull");
            e.printStackTrace();
            return false;
        }
    }


    public static List<Integer> findTop3Indices(float[] arr) {
        List<Integer> topIndices = new ArrayList<>();

        for (int i = 0; i < 3; i++) {
            int maxIndex = 0;
            float maxValue = arr[0];

            for (int j = 1; j < arr.length; j++) {
                if (arr[j] > maxValue && !topIndices.contains(j)) {
                    maxValue = arr[j];
                    maxIndex = j;
                }
            }

            topIndices.add(maxIndex);
        }

        return topIndices;
    }

    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Result<ImageClassificationResult> inference(Bitmap[] images, String backend) {
        System.out.println("Processing %d images %dx%d."+ images.length+ images[0].getWidth()+ images[0].getHeight());

        try {

            long Preprocessing_StartTime = System.nanoTime();
            Log.d(TAG,"Image Preprocessing");

            Utils util = new Utils();
            List<Float> img_mean = Arrays.asList(IMAGE_MEAN, IMAGE_MEAN, IMAGE_MEAN);
            List<Float> img_std = Arrays.asList(IMAGE_STD, IMAGE_STD, IMAGE_STD);

            int[] arr = tfLite.getInputTensor(0).shape(); //FOR VISION MODEL - input is normally like (B,H,W,C)
            int channel = arr[3];
            int input_dims1 = arr[1];
            int input_dims2 = arr[2];

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(images[0],input_dims1,input_dims2,true);

            float[][][][] floatinputarray = new float[1][input_dims1][input_dims1][channel];
            util.PreProcess(scaledBitmap, input_dims1, input_dims2, floatinputarray, img_mean, img_std);

            long Preprocessing_EndTime = System.nanoTime();
            long Preporccsing_TimeDiff=Preprocessing_EndTime-Preprocessing_StartTime;

            Log.d(TAG,"Preprocessing Time: "+Preporccsing_TimeDiff/1000000+"ms");

            Object[] inputArray = {floatinputarray};
            float[][] floatoutputarray = new float[1][1000];
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, floatoutputarray);

            long inferenceStartTime = System.nanoTime();

            if (backend.equals("NPU") && tfLite_QNN != null) {
                System.out.println("NPU BACKEND");
                tfLite_QNN.runForMultipleInputsOutputs(inputArray, outputMap);
            }
            else if (backend.equals("CPU") && tfLite != null) {
                System.out.println("TFLITE BACKEND");
                tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
            }
            else
            {
                System.out.println("Sycronisation issue");
            }

            Log.i(TAG, "MODEL EXECUTED");
            long inferenceEndTime = System.nanoTime();
            long TimeDiff=inferenceEndTime-inferenceStartTime;

            Toast.makeText(context,"Inference Time: "+TimeDiff/1000000+"ms",Toast.LENGTH_SHORT).show();
            Log.i(TAG,"Inference Completed");

            String res="";
            List<Integer> indexList = findTop3Indices(floatoutputarray[0]);

            for(int i=0;i<3;i++){
                res+=labelList.get(indexList.get(i)+1)+", ";
            }

            res = res.substring(0, res.length() - 2); //Removing comma from last

            ImageClassificationResult result = new ImageClassificationResult(indexList, res);

            return new Result<>(result,
                    (inferenceEndTime - inferenceStartTime) / 1000000);

        } catch (Exception ex) {
            ex.printStackTrace();
            return null;

        }
    }

    private List<String> loadLabelList(String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        AssetManager assetManager= context.getAssets();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
}
