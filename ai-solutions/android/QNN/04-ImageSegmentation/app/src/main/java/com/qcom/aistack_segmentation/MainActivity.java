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

import java.util.HashMap;
import java.util.Map;

/**
 * MainActivity class helps choose runtime from UI through main_activity.xml
 * Passes choice of runtime to CameraFragment for making inference using selected runtime.
 */
public class MainActivity extends AppCompatActivity {

    static Map<Character, String> runtime_name=new HashMap<Character, String>();

    static final String[] modeloptions = {"FCN_RESNET50", "FCN_RESNET100","DEEPLABV3_RESNET50","DEEPLABV3_RESNET101", "LRASPP"};

    static final char[] runtimeoptions = {'C', 'D'};

    Map<String, String> model_name = new HashMap<String, String>() {{
        put("libfcn_resnet50_quant16_w8a16.so", modeloptions[0]);
        put("fcn_resnet50_quant16_w8a16.serialized.bin", modeloptions[0]);
        put("libfcn_resnet101_quant16_w8a16.so", modeloptions[1]);
        put("fcn_resnet101_quant16_w8a16.serialized.bin", modeloptions[1]);
        put("libdeeplabv3_resnet50_quant_w8a8.so", modeloptions[2]);
        put("deeplabv3_resnet50_quant_w8a8.serialized.bin", modeloptions[2]);
        put("libdeeplabv3_resnet101_quant_w8a8.so", modeloptions[3]);
        put("deeplabv3_resnet101_quant_w8a8.serialized.bin", modeloptions[3]);
        put("liblraspp_w8a16.so", modeloptions[4]);
        put("lraspp_w8a16.serialized.bin", modeloptions[4]);
    }};

    Map<Character, String> backend = new HashMap<Character, String>() {{
        put(runtimeoptions[0], "libQnnCpu.so");
        put(runtimeoptions[1], "libQnnHtp.so");
    }};

    static final String[] runtimename = {"CPU", "DSP"};

    static {
        for (int i=0;i<runtimeoptions.length;i++) {
            runtime_name.put(runtimeoptions[i], runtimename[i]);
        }
    }

    public static String runtime_var; //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static Character runtime_char;

    public static String dlc_name;

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