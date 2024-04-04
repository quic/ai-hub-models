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

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.TextView;
import org.opencv.android.OpenCVLoader;
import java.util.HashMap;
import java.util.Map;

/**
 * MainActivity class displays the info of Selected Model,Runtime, classes supported by the model on UI through main_activity.xml
 * It also passes the selected model and runtime info to the CameraFragment for making inference using selected Model and Runtime.
 */
public class MainActivity extends AppCompatActivity {

    static final Map<String, String> model_name=new HashMap<String, String>();
    static Map<Character, String> runtime_name=new HashMap<Character, String>();
    static Map<String, String> class_count=new HashMap<String, String>();

    static final String[] modeloptions = {"YOLONAS", "SSDMobilenetV2", "YoloX"};
    static final String[] modeldlcname = {"Quant_yoloNas_s_320.dlc", "ssd_mobilenetV2_without_ABP-NMS_Q.dlc", "yolox_x_212_Q.dlc"};

    static final char[] runtimeoptions = {'C', 'G', 'D'};
    static final String[] runtimename = {"CPU", "GPU", "DSP"};

    static final String[] classcount = {"80", "21", "80"};

    static {
       //System.loadLibrary("objectdetectionYoloNas");

        for (int i=0;i<modeloptions.length;i++) {
            model_name.put(modeldlcname[i], modeloptions[i]);
        }

        for (int i=0;i<runtimeoptions.length;i++) {
            runtime_name.put(runtimeoptions[i], runtimename[i]);
        }

        for (int i=0;i<classcount.length;i++) {
            class_count.put(modeldlcname[i], classcount[i]);
        }

    }

    public static char runtime_var;

    public static String dlc_name;

    TextView tv1_val;
    TextView tv2_val;
    TextView tv3_val;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        OpenCVLoader.initDebug();


        Intent intentReceived = getIntent();
        Bundle data = intentReceived.getExtras();

        runtime_var = data.getChar("key");
        dlc_name = data.getString("selected_dlc_name");


        tv1_val=findViewById(R.id.tv1_val);
        tv2_val=findViewById(R.id.tv2_val);
        tv3_val=findViewById(R.id.tv3_val);

        tv1_val.setText(model_name.get(dlc_name));
        tv2_val.setText(runtime_name.get(runtime_var));
        tv3_val.setText(class_count.get(dlc_name));

        overToCamera(runtime_var, dlc_name);

    }

    /**
     * Method to request Camera permission
     */
    private void cameraPermission() {

        requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);

    }

    /**
     * Method to navigate to CameraFragment along with selected model dlc and runtime choice
     */
    private void overToCamera(char runtime_value, String selected_dlc_name) {
        Boolean passToFragment;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            passToFragment = MainActivity.this.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        }
        else{
            passToFragment = true;
        }
        if (passToFragment) {
            FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
            Bundle args = new Bundle();
            args.putChar("key", runtime_value);
            args.putCharSequence("selected_dlc_name", selected_dlc_name);
            transaction.add(R.id.main_content, CameraFragment.create(args));
            transaction.commit();
        } else {
            cameraPermission();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        overToCamera(runtime_var, dlc_name);
    }

    @Override
    protected void onStop() {
        super.onStop();
    }
}