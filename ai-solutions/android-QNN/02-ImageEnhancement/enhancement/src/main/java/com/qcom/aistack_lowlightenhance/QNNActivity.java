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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class QNNActivity extends AppCompatActivity {

    static {
        System.loadLibrary("ImageEnhancement");
        OpenCVLoader.initDebug();
    }

    QNNHelper mQnnHelper;
    Boolean mNetworkLoaded;
    float infer_time=0.0f;
    public static InputStream originalFile = null;

    //creating objects for UI element used in layout files (activity_qnn.xml)
    TextView txt_stat, tx_pr, tx_out, tx_sug;
    ImageView imageView, imageView2;
    RadioGroup radioGroup;
    Bitmap bmps = null;
    Bitmap outbmps = null;
    Spinner inputImageSpin;
    Spinner modelspin;
    String[] options = {"No Selection","Sample1.jpg","Sample2.jpg"}; //Image filenames on which model inference is made
    String[] modeloptions = {"No Selection", "RUAS", "SCI", "StableLLVE", "ZeroDCE"};
    String[] backendname = {"libQnnCpu.so", "libQnnHtp.so"};
    Map<String, String> runtimeSpecificModels = new HashMap<String, String>() {{
        put(modeloptions[1] + "|" + backendname[0], "libruas_w8a8.so");
        put(modeloptions[1] + "|" + backendname[1], "ruas_w8a8.serialized.bin");
        put(modeloptions[2] + "|" + backendname[0], "libsci_difficult_w8a8.so");
        put(modeloptions[2] + "|" + backendname[1], "sci_difficult_w8a8.serialized.bin");
        put(modeloptions[3] + "|" + backendname[0], "libStableLLVE_w8a8.so");
        put(modeloptions[3] + "|" + backendname[1], "StableLLVE_w8a8.serialized.bin");
        put(modeloptions[4] + "|" + backendname[0], "libzero_dce_w8a8.so");
        put(modeloptions[4] + "|" + backendname[1], "zero_dce_w8a8.serialized.bin");
    }};
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
                infer_time = 0.0f;
                status = process(bmps,"libQnnCpu.so",
                        runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnCpu.so"));
                timestr = "CPU inference time : " + infer_time + "milli sec";

                break;
            case R.id.rb3:
                System.out.println("DSP instance running");
                infer_time = 0.0f;

                status = process(bmps,"libQnnHtp.so",
                        runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnHtp.so"));
                    timestr = "DSP Inference time : " + infer_time + "milli sec";
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
                            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            if (final_status == true) {
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
                        imageView2.setVisibility(view.INVISIBLE);
                        System.out.println("MotionEvent.ACTION_DOWN");
                        tx_out.setVisibility(view.INVISIBLE);
                        tx_pr.setVisibility(view.VISIBLE);
                        break;
                    }
                    case MotionEvent.ACTION_UP: {
                        imageView2.setVisibility(view.VISIBLE);
                        System.out.println("MotionEvent.ACTION_UP");
                        tx_out.setVisibility(view.VISIBLE);
                        tx_pr.setVisibility(view.INVISIBLE);
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

                // loading picture from assets...
                if (!parent.getItemAtPosition(position).equals("No Selection") && !inputImageSpin.getSelectedItem().toString().equals("No Selection")) {//if no selection of image
                    txt_stat.setText("Stats");
                    try {
                        originalFile = getAssets().open(inputImageSpin.getSelectedItem().toString());
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
                        originalFile = getAssets().open(inputImageSpin.getSelectedItem().toString());
                        // Set the input image in UI view
                        imageView.setImageBitmap(BitmapFactory.decodeStream(originalFile));
                        imageView2.setImageResource(R.drawable.ic_launcher_background);
                        imageView2.setVisibility(view.INVISIBLE);

                    } catch (Exception e) {
                        e.printStackTrace();
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

                // loading picture from assets...
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
                        originalFile = getAssets().open(inputImageSpin.getSelectedItem().toString());
                        imageView.setImageBitmap(BitmapFactory.decodeStream(originalFile));
                        imageView2.setImageResource(R.drawable.ic_launcher_background);
                        imageView2.setVisibility(view.INVISIBLE);
                    } catch (Exception e) {
                        e.printStackTrace();
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
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                System.out.println("Nothing");
            }
        });
    }

    //Function to load model and get inference from it
    public boolean process(Bitmap bmps, String runtime_var, String model_name) {

        mQnnHelper = new QNNHelper(getApplication());

        mNetworkLoaded = mQnnHelper.loadingMODELS(runtime_var, model_name);

        if (mNetworkLoaded == true)
        {
            outbmps = mQnnHelper.qnnInference(bmps);
            infer_time = mQnnHelper.getInfer_time();
        }

        if (outbmps == null)
        {
            System.out.println("outbmps is null");
                return false;
        }
        return true;
    }
}

