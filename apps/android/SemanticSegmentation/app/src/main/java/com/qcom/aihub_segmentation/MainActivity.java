// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.aihub_segmentation;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentTransaction;

public class MainActivity extends AppCompatActivity {

    public static char runtime_var;
    private static final String TAG = "Segmentation_MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        Intent intentReceived = getIntent();
        Bundle data = intentReceived.getExtras();

        runtime_var = data.getChar("key");
        Log.d(TAG,"Main Activity runtimevar"+runtime_var);

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
    private void overToCamera(char runtime_value) {
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
            transaction.add(R.id.main_content, CameraFragment.create(args));
            transaction.commit();
        } else {
            cameraPermission();
        }
    }

    @Override
    protected void onResume()
    {
        super.onResume();
        overToCamera(runtime_var);
    }
    @Override
    protected void onStop() {
        super.onStop();
    }
}
