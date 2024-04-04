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

import java.util.HashMap;
import java.util.Map;


/**
 * HomeScreenActivity class helps choose runtime from UI through activity_home_screen.xml
 * Passes choice of selected Model and runtime to MainActivity.
 */
public class HomeScreenActivity extends AppCompatActivity{

    static {

        runtime_var = 'C';
        dlc_name = "libyolo_nas_w8a8.so";

        System.loadLibrary("objectdetectionYoloNas");
    }

    public static char runtime_var;  //TODO change here as well as main_activity.xml, change checked "android:checked="true""

    public static String dlc_name;

    RadioGroup rg;

    Spinner modelspin;

    String[] modeloptions = {"YOLONAS", "YoloX"};

    String[] backendname = {"libQnnCpu.so", "libQnnHtp.so"};

    Map<String, String> runtimeSpecificModels = new HashMap<String, String>() {{
        put(modeloptions[0] + "|" + backendname[0], "libyolo_nas_w8a8.so");
        put(modeloptions[0] + "|" + backendname[1], "yolo_nas_w8a8.serialized.bin");

        put(modeloptions[1] + "|" + backendname[0], "libyolox_a8w8_2_15_1.so");
        put(modeloptions[1] + "|" + backendname[1], "yolox_a8w8_2_15_1.serialized.bin");
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
                //Toast.makeText(getApplicationContext(), modeloptions[position] + " selected.", Toast.LENGTH_SHORT).show();
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
