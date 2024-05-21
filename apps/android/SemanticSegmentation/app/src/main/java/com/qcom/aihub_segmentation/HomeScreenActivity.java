// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.aihub_segmentation;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.RadioGroup;
import androidx.appcompat.app.AppCompatActivity;
import java.io.IOException;

public class HomeScreenActivity extends AppCompatActivity {

    public static char runtime_var='C';
    private TFLiteModelExecutor inferObj;
    private final static String TFLITE_FILE = "segmentationModel.tflite";
    RadioGroup rg;
    private static final String TAG="Segmentation_HomeScreen";

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home_screen);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        rg = (RadioGroup) findViewById(R.id.rg1);
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                switch (checkedId) {
                    case R.id.CPU:
                        runtime_var = 'C';
                        Log.d(TAG,"CPU instance selected.");
                        break;
                    case R.id.NPU:
                        runtime_var = 'N';
                        Log.d(TAG,"NPU instance selected.");
                        break;
                    default:
                        runtime_var = 'C';
                        Log.d(TAG,"Nothing selected.");
                        break;
                }
            }
        });

        inferObj = new TFLiteModelExecutor();
        createNetwork();
    }

    private void createNetwork() {
            ModelManager infermgr = ModelManager.getInstance();
            infermgr.initializeModelExecutor(inferObj);
            new Thread() {
                public void run() {

                    try {
                        inferObj.initializingModel(getApplicationContext(),TFLITE_FILE);
                    } catch (IOException e) {
                        Log.e(TAG,"Model is not loaded Successfully");
                        throw new RuntimeException(e);
                    }
                }
            }.start() ;
    }

    public void startManinCameraActivity(View v)
    {
        Intent i =new Intent(this, MainActivity.class);

        Bundle args = new Bundle();
        args.putChar("key", runtime_var);
        i.putExtras(args);
        startActivity(i);
    }
}
