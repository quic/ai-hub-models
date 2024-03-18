// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imageclassification;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;

public class QNNActivity extends AppCompatActivity {

    public static InputStream originalFile = null;
    ImageClassification imageClassification;
    private final static String TFLITE_FILE = "classification.tflite";

    //creating objects for UI element used in layout files (activity_classification.xml)
    RadioButton rb1, rb2, rb3;

    String prev_runtime = "";
    ImageView imageView;
    RadioGroup radioGroup;
    TextView predicted_view;
    Bitmap bmps = null;
    private boolean spinInitialized = false;
    private boolean radioGroupInitialized = false;
    public static Result<ImageClassificationResult> result = null;
    Spinner spin;
    private static final String TAG="Image_Classification";

      String[] options = {"No Selection","Sample1.png","Sample2.png","Sample3.png"}; //Image filenames on which model inference is made
    protected void executeRadioButton(int checkedId) {
        switch (checkedId) {
            case R.id.rb1:
                // set text for your textview here
                System.out.println("CPU instance running");
                result = process(bmps, "CPU");
                break;
            case R.id.rb2:
                // set text for your textview here
                System.out.println("NPU instance running");
                System.out.println("Device runtime " + "NPU");
                result = process(bmps, "NPU");
                break;
            default:
                System.out.println("Do Nothing");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //Initialization
        setContentView(R.layout.activity_classification);
        rb1 = (RadioButton) findViewById(R.id.rb1);
        rb2 = (RadioButton) findViewById(R.id.rb2);
        imageView = (ImageView) findViewById(R.id.im1);
        radioGroup = (RadioGroup) findViewById(R.id.rg1);
        spin = (Spinner) findViewById((R.id.spinner));
        predicted_view=(TextView)findViewById(R.id.textView4);

        predicted_view.setVisibility(View.INVISIBLE);

        imageClassification = new ImageClassification();

        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, options);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spin.setAdapter(ad);


        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                if (originalFile!=null && bmps!=null){
                    executeRadioButton(checkedId);
                }
                else{
                    if(radioGroupInitialized) {
                        Toast.makeText(getApplicationContext(), "Please select image first", Toast.LENGTH_SHORT).show();
                    }
                    else
                    {
                        radioGroupInitialized = true;
                    }
                }
            }
        });

        spin.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

                // loading picture from assets...
                if (!parent.getItemAtPosition(position).equals("No Selection")) {
                    try {
                        originalFile = getAssets().open((String) parent.getItemAtPosition(position));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Convert input image to Bitmap
                    bmps = BitmapFactory.decodeStream(originalFile);

                    //Scaling the image size to show it on the ImageView
                    Bitmap scaled1 = Bitmap.createScaledBitmap(bmps, 512, 512, true);
                    try {
                        // Set the input image in UI view
                        imageView.setImageBitmap(scaled1);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    //Taking the Runtime Environment input from Radio Button
                    int checkedID_RB = radioGroup.getCheckedRadioButtonId();
                    if (originalFile!=null && bmps!=null && checkedID_RB !=-1){
                        executeRadioButton(checkedID_RB);
                    }
                }
                else{

                    originalFile=null;
                    bmps=null;
                    imageView.setImageResource(R.drawable.ic_launcher_background);
                    radioGroup.clearCheck();

                    if(spinInitialized){
                        Toast.makeText(getApplicationContext(), "Please select image first", Toast.LENGTH_SHORT).show();
                    }
                    else
                    {
                        spinInitialized = true;
                    }
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                System.out.println("Nothing");
            }
        });
    }

    public Result<ImageClassificationResult> process(Bitmap bmps, String run_time){

        Result<ImageClassificationResult> result = null;
        try {
        if(imageClassification.getBuildStatus()==false)
        		imageClassification.initializeModel(this, TFLITE_FILE);

            result = imageClassification.inference(new Bitmap[] {bmps}, run_time);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        predicted_view.setVisibility(View.VISIBLE);
        predicted_view.setText(result.getResults().getResultString());
        return result;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        imageClassification.close();
    }
}
