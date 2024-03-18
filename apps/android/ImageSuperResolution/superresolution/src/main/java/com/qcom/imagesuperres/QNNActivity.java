// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;

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
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class QNNActivity extends AppCompatActivity {
    public static final String MODEL_FILE_NAME = "superresmodel.tflite"; //Model file name
    public static InputStream originalFile = null;

    private boolean spinInitialized = false;
    private boolean radioGroupInitialized = false;
    SuperResolution superResolution;

    String prev_runtime = "";
    //creating objects for UI element used in layout files (activity_superres.xml)
    TextView txt_stat, tx_pr, tx_out, tx_sug;
    private static int input_dims1 = 128;
    private static int  input_dims2 = 128;
    ImageView imageView, imageView2;
    RadioGroup radioGroup;
    Bitmap bmps = null;
    public static Result<SuperResolutionResult> result = null;
    Spinner spin;
    String[] options = {"No Selection","Sample1.jpg","Sample2.jpg"}; //Image filenames on which model inference is made
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
                    switch (checkedId) {
                        case R.id.rb1:
                            // set text for your textview here
                            System.out.println("CPU instance running");
                            result = process(bmps, "TFLITE");
                            break;

                        case R.id.rb3:
                            System.out.println("NPU instance running");
                            result = process(bmps, "QNNDELEGATE");
                            break;
                        default:
                            System.out.println("Do Nothing");
                    }
                    boolean final_status = result.getStatus();
                    final String final_timestr = "INFERENCE TIME: "+ String.valueOf(result.getInferenceTime())+" ms";
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            txt_stat.setText(final_timestr);
                            progressBar.setVisibility(View.INVISIBLE);
                            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            if (final_status == true) {
                                String remark = result.getRemarks();
                                if(!remark.equals(""))
                                    Toast.makeText(getApplicationContext(),remark,Toast.LENGTH_LONG).show();
                                imageView2.setImageBitmap(result.getResults().getHighResolutionImages()[0]);
                                imageView2.setVisibility(View.VISIBLE);
                                System.out.println("result displayed");
                                txt_stat.setVisibility(View.VISIBLE);
                                tx_pr.setVisibility(View.INVISIBLE);
                                tx_out.setVisibility(View.VISIBLE);
                                tx_sug.setVisibility(View.VISIBLE);
                            }
                        }
                    });
                } catch (Exception e) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    e.printStackTrace();
                        }
                    });
                }
            }
        });
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_superres);
        spin = (Spinner) findViewById((R.id.spinner));
        txt_stat = findViewById(R.id.textView4);
        imageView = findViewById(R.id.im1);
        imageView2 = findViewById(R.id.im2);
        radioGroup = findViewById(R.id.rg1);
        tx_pr = findViewById(R.id.textView);
        tx_out = findViewById(R.id.textView2);
        tx_sug = findViewById(R.id.textView_suggest);
        imageView2.setVisibility(View.INVISIBLE);
        tx_out.setVisibility(View.INVISIBLE);
        tx_sug.setVisibility(View.INVISIBLE);

        superResolution = new SuperResolution();

        imageView2.setOnTouchListener((view, motionEvent) -> {
            switch (motionEvent.getAction()) {
                case MotionEvent.ACTION_DOWN: {
                    imageView2.setVisibility(view.INVISIBLE);
                    tx_out.setVisibility(view.INVISIBLE);
                    tx_pr.setVisibility(view.VISIBLE);
                    break;
                }
                case MotionEvent.ACTION_UP: {
                    imageView2.setVisibility(view.VISIBLE);
                    tx_out.setVisibility(view.VISIBLE);
                    tx_pr.setVisibility(view.INVISIBLE);
                    break;
                }
            }
            return false;
        });

        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, options);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spin.setAdapter(ad);
        spin.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // loading picture from assets...
                if (!parent.getItemAtPosition(position).equals("No Selection")) {
                    imageView2.setImageResource(R.drawable.ic_launcher_background);
                    txt_stat.setText("Stats");
                    try {
                        originalFile = getAssets().open((String) parent.getItemAtPosition(position));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Convert input image to Bitmap
                    bmps = BitmapFactory.decodeStream(originalFile);
                    Bitmap scaled1 = Bitmap.createScaledBitmap(bmps, input_dims1, input_dims2, true);
                    try {
                        // Set the input image in UI view
                        imageView.setImageBitmap(scaled1);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    int checkedID_RB = radioGroup.getCheckedRadioButtonId();
                    if (originalFile!=null && bmps!=null && checkedID_RB !=-1){
                        executeRadioButton(checkedID_RB);
                    }

                }
                else{
                    originalFile=null;
                    bmps=null;
                    imageView.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setImageResource(R.drawable.ic_launcher_background);
                    imageView2.setVisibility(view.INVISIBLE);
                    txt_stat.setText("Stats");
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
    }

    public Result<SuperResolutionResult> process(Bitmap bmps, String run_time) {

        Result<SuperResolutionResult> result;
        try {

            if(superResolution.getBuildStatus()==false)
                superResolution.initializeModel(this, MODEL_FILE_NAME);

            //INFERENCING ON MODEL
            result = superResolution.inference(new Bitmap[]{bmps}, run_time);
            return result;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }

    @Override
    protected void onDestroy()
    {
        super.onDestroy();
        superResolution.close();
    }
}
