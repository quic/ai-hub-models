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

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Toast;

public class HomeScreenActivity extends AppCompatActivity {

    static {
        System.loadLibrary("imageSegmentation");
    }

    public static char runtime_var;  //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static String dlc_name;

    RadioGroup rg;

    Spinner modelspin;
    String[] modeloptions = {"FCN_RESNET50", "FCN_RESNET100","DEEPLABV3_RESNET50","DEEPLABV3_RESNET101","LRASPP"};
    String[] modeldlcname = {"fcn_resnet50_quant16_w8a16.dlc","fcn_resnet101_quant16_w8a16.dlc","deeplabv3_resnet50_quant_w8a8.dlc","deeplabv3_resnet101_quant_w8a8.dlc","lraspp_mobilenet_v3_large_quant16_w8a16.dlc"};


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
                        System.out.println("sanjeev CPU instance running dlc_name ="+modeldlcname[modelspin.getSelectedItemPosition()]);
                        break;
                    case R.id.GPU:
                        runtime_var = 'G';
                        System.out.println("sanjeev GPU instance running dlc_name ="+modeldlcname[modelspin.getSelectedItemPosition()]);
                        break;
                    case R.id.DSP:
                        runtime_var = 'D';
                        System.out.println("sanjeev DSP instance running dlc_name ="+modeldlcname[modelspin.getSelectedItemPosition()]);
                        break;
                    default:
                        runtime_var = 'N';
                        System.out.println("sanjeev Nothing selected dlc_name ="+dlc_name);
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

                Toast.makeText(getApplicationContext(), modeloptions[position] + " selected.", Toast.LENGTH_SHORT).show();

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
        args.putCharSequence("selected_dlc_name", modeldlcname[modelspin.getSelectedItemPosition()]);
        i.putExtras(args);

        startActivity(i);
    }

}