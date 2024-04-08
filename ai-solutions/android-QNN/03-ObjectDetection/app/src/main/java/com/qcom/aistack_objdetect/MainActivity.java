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

    static Map<Character, String> runtime_name=new HashMap<Character, String>();

    static final String[] modeloptions = {"YOLONAS", "YoloX"};

    static final char[] runtimeoptions = {'C', 'D'};

    Map<String, String> model_name = new HashMap<String, String>() {{
        put("libyolo_nas_w8a8.so", modeloptions[0]);
        put("yolo_nas_w8a8.serialized.bin", modeloptions[0]);
        put("libyolox_a8w8_2_15_1.so", modeloptions[1]);
        put("yolox_a8w8_2_15_1.serialized.bin", modeloptions[1]);
    }};

    Map<Character, String> backend = new HashMap<Character, String>() {{
        put(runtimeoptions[0], "libQnnCpu.so");
        put(runtimeoptions[1], "libQnnHtp.so");
    }};

    static final String[] runtimename = {"CPU", "DSP"};

    static final String[] classcount = {"80", "80"};

    Map<String, String> class_count=new HashMap<String, String>() {{
        put("libyolo_nas_w8a8.so", classcount[0]);
        put("yolo_nas_w8a8.serialized.bin", classcount[0]);
        put("libyolox_a8w8_2_15_1.so", classcount[1]);
        put("yolox_a8w8_2_15_1.serialized.bin", classcount[1]);
    }};

    static {
       //System.loadLibrary("objectdetectionYoloNas");

        for (int i=0;i<runtimeoptions.length;i++) {
            runtime_name.put(runtimeoptions[i], runtimename[i]);
        }
    }

    public static String runtime_var;

    public static Character runtime_char;

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

        runtime_char = data.getChar("key");
        System.out.println("runtime_char: " + runtime_char);
        runtime_var = backend.get(runtime_char);
        System.out.println("runtime_var: " + runtime_var);
        dlc_name = data.getString("selected_dlc_name");

        tv1_val=findViewById(R.id.tv1_val);
        tv2_val=findViewById(R.id.tv2_val);
        tv3_val=findViewById(R.id.tv3_val);

        tv1_val.setText(model_name.get(dlc_name));
        tv2_val.setText(runtime_name.get(runtime_char));
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
    private void overToCamera(String runtime_value, String selected_dlc_name) {
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
            args.putCharSequence("key", runtime_value);
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