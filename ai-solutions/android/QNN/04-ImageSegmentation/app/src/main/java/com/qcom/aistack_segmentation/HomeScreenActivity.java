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

import java.util.HashMap;
import java.util.Map;

public class HomeScreenActivity extends AppCompatActivity {

    static {
        runtime_var = 'C';
        dlc_name = "libfcn_resnet50_quant16_w8a16.so";

        System.loadLibrary("imageSegmentation");
    }

    public static char runtime_var;  //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static String dlc_name;

    RadioGroup rg;

    Spinner modelspin;
//    String[] modeloptions = {"FCN_RESNET50", "FCN_RESNET100","DEEPLABV3_RESNET50","DEEPLABV3_RESNET101","LRASPP"};
    String[] modeloptions = {"FCN_RESNET50", "FCN_RESNET100","DEEPLABV3_RESNET50","DEEPLABV3_RESNET101", "LRASPP"};
    String[] backendname = {"libQnnCpu.so", "libQnnHtp.so"};

    Map<String, String> runtimeSpecificModels = new HashMap<String, String>() {{
        put(modeloptions[0] + "|" + backendname[0], "libfcn_resnet50_quant16_w8a16.so");
        put(modeloptions[0] + "|" + backendname[1], "fcn_resnet50_quant16_w8a16.serialized.bin");

        put(modeloptions[1] + "|" + backendname[0], "libfcn_resnet101_quant16_w8a16.so");
        put(modeloptions[1] + "|" + backendname[1], "fcn_resnet101_quant16_w8a16.serialized.bin");

        put(modeloptions[2] + "|" + backendname[0], "libdeeplabv3_resnet50_quant_w8a8.so");
        put(modeloptions[2] + "|" + backendname[1], "deeplabv3_resnet50_quant_w8a8.serialized.bin");

        put(modeloptions[3] + "|" + backendname[0], "libdeeplabv3_resnet101_quant_w8a8.so");
        put(modeloptions[3] + "|" + backendname[1], "deeplabv3_resnet101_quant_w8a8.serialized.bin");

        put(modeloptions[4] + "|" + backendname[0], "liblraspp_w8a16.so");
        put(modeloptions[4] + "|" + backendname[1], "lraspp_w8a16.serialized.bin");
    }};



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
                        dlc_name = runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnCpu.so");
                        System.out.println("CPU instance selected dlc_name ="+dlc_name);
                        break;
                    case R.id.DSP:
                        runtime_var = 'D';
                        dlc_name = runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnHtp.so");
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

//                Toast.makeText(getApplicationContext(), modeloptions[position] + " selected.", Toast.LENGTH_SHORT).show();
                if(runtime_var == 'D')
                    dlc_name = runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnHtp.so");
                else
                    dlc_name = runtimeSpecificModels.get(modeloptions[modelspin.getSelectedItemPosition()] + "|libQnnCpu.so");
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