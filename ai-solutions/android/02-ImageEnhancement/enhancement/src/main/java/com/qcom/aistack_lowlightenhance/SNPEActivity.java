// -*- mode: java -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
package com.qcom.aistack_lowlightenhance;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SNPEActivity extends AppCompatActivity {

    static {
        System.loadLibrary("ImageEnhancement");
        OpenCVLoader.initDebug();
    }

    SNPEHelper mSnpeHelper;
    Boolean mNetworkLoaded;

    float infer_time=0.0f;

    public static InputStream originalFile = null;

    //creating objects for UI element used in layout files (activity_snpe.xml)
    TextView txt_stat, tx_pr, tx_out, tx_sug;
    ImageView imageView, imageView2;
    RadioGroup radioGroup;
    Bitmap bmps = null;
    Bitmap outbmps = null;
    Spinner inputImageSpin;
    Spinner modelspin;
    final String[] options = {"No Selection","Sample1.jpg","Sample2.jpg"}; //Image filenames on which model inference is made
    final String[] modeldlcname = {"None", "ruas_Q.dlc", "sci_difficult_Q.dlc", "StableLLVE_Q.dlc", "quant_zeroDCE_640_480_212_8550_out80.dlc"};
    final String[] modeloptions = { "No Selection", "RUAS", "SCI", "StableLLVE", "ZeroDCE"};
    protected void executeRadioButton(int checkedId) {

        ProgressBar progressBar;
        progressBar = findViewById(R.id.indeterminateBar);
        ExecutorService service = Executors.newSingleThreadExecutor();
        progressBar.setVisibility(View.VISIBLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);

        service.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    boolean status = false;
                    String timestr = null;
                    switch (checkedId) {
                        case R.id.rb1:
                            // set text for your textview here
                            System.out.println("CPU instance running");

                            status = process(bmps, 'C', modeldlcname[modelspin.getSelectedItemPosition()]);
                            timestr = "CPU inference time : " + infer_time + " ms";
                            break;

                        case R.id.rb2:
                            // set text for your textview here
                            System.out.println("GPU instance running");

                            status = process(bmps, 'G', modeldlcname[modelspin.getSelectedItemPosition()]);
                            timestr = "GPU inference time : " + infer_time + " ms";
                            break;

                        case R.id.rb3:
                            System.out.println("DSP instance running");

                            status = process(bmps, 'D', modeldlcname[modelspin.getSelectedItemPosition()]);
                            timestr = "DSP Inference time : " + infer_time + "ms";
                            break;

                        default:
                            System.out.println("Do Nothing");
                            break;

                    }

                    boolean final_status = status;
                    final String final_timestr = timestr;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            txt_stat.setText(final_timestr);
                            progressBar.setVisibility(View.INVISIBLE);

                            //making UI responsive
                            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);

                            if (final_status) {
                                imageView2.setImageBitmap(outbmps);
                                imageView2.setVisibility(View.VISIBLE);
                                txt_stat.setVisibility(View.VISIBLE);
                                tx_pr.setVisibility(View.INVISIBLE);
                                tx_out.setVisibility(View.VISIBLE);
                                tx_sug.setVisibility(View.VISIBLE);
                            }
                        }
                    });
                }
                catch(Exception e)
                {
                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    e.printStackTrace();
                }
            }

        });
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_snpe);
        txt_stat = findViewById(R.id.textView4);
        imageView = findViewById(R.id.im1);
        imageView2 = findViewById(R.id.im2);
        radioGroup = findViewById(R.id.rg1);
        inputImageSpin = findViewById((R.id.spinner));
        modelspin = findViewById((R.id.spinner7));
        tx_pr = findViewById(R.id.textView);
        tx_out = findViewById(R.id.textView2);
        tx_sug = findViewById(R.id.textView_suggest);
        imageView2.setVisibility(View.INVISIBLE);
        tx_out.setVisibility(View.INVISIBLE);
        tx_sug.setVisibility(View.INVISIBLE);


        imageView2.setOnTouchListener((view, motionEvent) -> {
            switch (motionEvent.getAction()) {
                case MotionEvent.ACTION_DOWN: {
                    imageView2.setVisibility(View.INVISIBLE);
                    System.out.println("MotionEvent.ACTION_DOWN");
                    tx_out.setVisibility(View.INVISIBLE);
                    tx_pr.setVisibility(View.VISIBLE);
                    break;
                }
                case MotionEvent.ACTION_UP: {
                    imageView2.setVisibility(View.VISIBLE);
                    System.out.println("MotionEvent.ACTION_UP");
                    tx_out.setVisibility(View.VISIBLE);
                    tx_pr.setVisibility(View.INVISIBLE);
                    break;
                }
            }
            return false;
        });

        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, options);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        inputImageSpin.setAdapter(ad);

        ArrayAdapter ad7 = new ArrayAdapter(this, android.R.layout.simple_spinner_item, modeloptions);
        ad7.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelspin.setAdapter(ad7);

        // Listener to check the change in HW accelerator input in APP UI
        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (!inputImageSpin.getSelectedItem().toString().equals("No Selection") && !modelspin.getSelectedItem().toString().equals("No Selection")){
                    executeRadioButton(checkedId);
                }
                else if (checkedId!=-1 && inputImageSpin.getSelectedItem().toString().equals("No Selection") && modelspin.getSelectedItem().toString().equals("No Selection")){
                    Toast.makeText(getApplicationContext(), "Please select model and image", Toast.LENGTH_SHORT).show();
                }
                else if (checkedId!=-1  && inputImageSpin.getSelectedItem().toString().equals("No Selection"))
                {
                    Toast.makeText(getApplicationContext(), "Please select image to model ", Toast.LENGTH_SHORT).show();
                }
                else if(checkedId!=-1  && modelspin.getSelectedItem().toString().equals("No Selection"))
                {
                    Toast.makeText(getApplicationContext(), "Please select appropriate model ", Toast.LENGTH_SHORT).show();
                }
            }
        });

        modelspin.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

                //Model and Runtime is mentioned
                if (!parent.getItemAtPosition(position).equals("No Selection") && !inputImageSpin.getSelectedItem().toString().equals("No Selection")) {//if no selection of image
                    txt_stat.setText("Stats");

                    //Loading assets
                    try {
                        originalFile = getAssets().open((String) inputImageSpin.getSelectedItem().toString());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Convert input image to Bitmap
                    bmps = BitmapFactory.decodeStream(originalFile);
                    try {
                        // Set the input image in UI view
                        imageView.setImageBitmap(bmps);
                        System.out.println("modelspin: INPUT wxh:"+bmps.getWidth()+"-----"+bmps.getHeight());
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    int checkedID_RB = radioGroup.getCheckedRadioButtonId();
                    if (originalFile!=null && bmps!=null && checkedID_RB !=-1){
                        executeRadioButton(checkedID_RB);
                    }
                }
                else if (!inputImageSpin.getSelectedItem().toString().equals("No Selection")) {

                    try {
                        originalFile = getAssets().open((String) inputImageSpin.getSelectedItem().toString());
                        // Set the input image in UI view
                        imageView.setImageBitmap(BitmapFactory.decodeStream(originalFile));
                        imageView2.setImageResource(R.drawable.ic_launcher_background);
                        imageView2.setVisibility(View.INVISIBLE);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }
                else{
                    originalFile=null;
                    bmps=null;
                    imageView.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setVisibility(View.INVISIBLE);
                    txt_stat.setText("Stats");
                    radioGroup.clearCheck();
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                System.out.println("Nothing");
            }
        });

        inputImageSpin.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

                //If model and image is selected
                if (!parent.getItemAtPosition(position).equals("No Selection") && !modelspin.getSelectedItem().toString().equals("No Selection")) {//if no selection of image
                    txt_stat.setText("Stats");
                    try {
                        // loading picture from assets...
                        originalFile = getAssets().open((String) parent.getItemAtPosition(position));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Convert input image to Bitmap
                    bmps = BitmapFactory.decodeStream(originalFile);
                    try {
                        // Set the input image in UI view
                        imageView.setImageBitmap(bmps);

                        System.out.println("INPUT wxh: "+bmps.getWidth()+"-----"+bmps.getHeight());
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    int checkedID_RB = radioGroup.getCheckedRadioButtonId();
                    if (originalFile!=null && bmps!=null && checkedID_RB !=-1){
                        executeRadioButton(checkedID_RB);
                    }
                }
                //if only input image is selected
                else if (!inputImageSpin.getSelectedItem().toString().equals("No Selection")) {

                    try {
                        originalFile = getAssets().open((String) inputImageSpin.getSelectedItem().toString());

                        // Set the input image in UI view
                        imageView.setImageBitmap(BitmapFactory.decodeStream(originalFile));
                        imageView2.setImageResource(R.drawable.ic_launcher_background);
                        imageView2.setVisibility(View.INVISIBLE);


                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                else{
                    originalFile=null;
                    bmps=null;
                    imageView.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setVisibility(View.INVISIBLE);
                    txt_stat.setText("Stats");
                    radioGroup.clearCheck();
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                System.out.println("Nothing");
            }
        });
    }

    //Function to load model and get inference from it
    public boolean process(Bitmap bmps, char runtime_var, String dlc_name) {

        mSnpeHelper = new SNPEHelper(getApplication());

        mNetworkLoaded = mSnpeHelper.loadingMODELS(runtime_var, dlc_name);

        if (mNetworkLoaded)
        {
            outbmps = mSnpeHelper.snpeInference(bmps);
            infer_time = mSnpeHelper.getInfer_time();
        }

        if (outbmps == null)
        {
            System.out.println("outbmps is null");
            return false;
        }
        return true;
    }
}

