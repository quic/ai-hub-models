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

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;

import org.opencv.android.OpenCVLoader;

/**
 * MainActivity class helps choose runtime from UI through main_activity.xml
 * Passes choice of runtime to CameraFragment for making inference using selected runtime.
 */
public class MainActivity extends AppCompatActivity {


    public static char runtime_var;  //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static String dlc_name;

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

        overToCamera(runtime_var, dlc_name);

    }

    /**
     * Method to request Camera permission
     */
    private void cameraPermission() {
        requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
    }

    /**
     * Method to navigate to CameraFragment along with runtime choice
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