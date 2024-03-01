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

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.RadioGroup;
import android.widget.Spinner;


/**
 * HomeScreenActivity class helps choose runtime from UI through activity_home_screen.xml
 * Passes choice of selected Model and runtime to MainActivity.
 */
public class HomeScreenActivity extends AppCompatActivity{

    static {

        runtime_var = 'C';
        dlc_name = "Quant_yoloNas_s_320.dlc";

        System.loadLibrary("objectdetectionYoloNas");
    }

    public static char runtime_var;  //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static String dlc_name;

    RadioGroup rg;

    Spinner modelspin;

    String[] modeloptions = {"YOLONAS", "SSDMobilenetV2", "YoloX"};
    String[] modeldlcname = {"Quant_yoloNas_s_320.dlc", "ssd_mobilenetV2_without_ABP-NMS_Q.dlc", "yolox_x_212_Q.dlc"};


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
                        System.out.println("CPU instance selected dlc_name ="+dlc_name);
                        break;
                    case R.id.GPU:
                        runtime_var = 'G';
                        System.out.println("GPU instance selected dlc_name ="+dlc_name);
                        break;
                    case R.id.DSP:
                        runtime_var = 'D';
                        System.out.println("DSP instance selected dlc_name ="+dlc_name);
                        break;
                    default:
                        runtime_var = 'N';
                        System.out.println("Nothing selected dlc_name ="+dlc_name);
                }
            }
        });

        modelspin = (Spinner)findViewById((R.id.spinnerNew1));

        ArrayAdapter adNew1 = new ArrayAdapter(this, android.R.layout.simple_spinner_item, modeloptions);
        adNew1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelspin.setAdapter(adNew1);


        modelspin.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                //Toast.makeText(getApplicationContext(), modeloptions[position] + " selected.", Toast.LENGTH_SHORT).show();
                dlc_name = modeldlcname[modelspin.getSelectedItemPosition()];
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                System.out.println("Nothing");
            }
        });

    }

    public void startManinCameraActivity(View v)
    {
        Intent i =new Intent(this, MainActivity.class);

        Bundle args = new Bundle();
        args.putChar("key", runtime_var);
        args.putCharSequence("selected_dlc_name", dlc_name);
        i.putExtras(args);

        startActivity(i);
    }

}
