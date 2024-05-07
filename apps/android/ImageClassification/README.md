### Requirements

1. Java, android-sdk and sdkmanager is already set at user's end
2. User should have Linux QNN SDK in local machine.
3. If models need to be downloaded during the execution of build_apk.py, qai-hub must be installed first according to the[ README.md in ai-hub-models](https://github.com/quic/ai-hub-models/blob/main/README.md).

4. When downloading [MobileNet_v3_Small](https://aihub.qualcomm.com/mobile/models/mobilenet_v3_small) via the web interface, qai-hub installation can be temporarily skipped.



## Info
Right now we use mobilenet_v3_small.tflite model which takes 224x224 as input and gives array of 1000 as output. You can replace it with any tflite classification model, but you have to change the pre-processing, post-processing and dimensions in the app code based on model parameters.


## Preprocessing


```
    for (int x = 0; x < input_dims1; x++) {
        for (int y = 0; y < input_dims2; y++) {
            int pixel = inputBitmap.getPixel(x, y);
            List<Float> rgb = Arrays.asList((float)Color.red(pixel), (float)Color.green(pixel), (float)Color.blue(pixel));
            for(int z = 0;z<3; z++){
                floatinputarray[0][z][x][y] = (float)((rgb.get(z))-ImageMean.get(z))/ImageStd.get(z);
            }
        }
    }
```


## PostProcessing


```
    public static List<Integer> findTop3Indices(float[] arr) {
    List<Integer> topIndices = new ArrayList<>();

    for (int i = 0; i < 3; i++) {
        int maxIndex = 0;
        float maxValue = arr[0];

        for (int j = 1; j < arr.length; j++) {
            if (arr[j] > maxValue && !topIndices.contains(j)) {
                maxValue = arr[j];
                maxIndex = j;
            }
        }

        topIndices.add(maxIndex);
    }

    return topIndices;
    }
```

### Build App:

You have to run build_apk.py for Image Classification. It will generate classification-debug.apk and install it in connected device.

Please first use `python build_apk.py -h` to understand the parameters.


```
build_apk.py [-h] -q QNNSDK [-m MODEL_NAME] [-path MODEL_PATH]
```

The parameter "-q" must be entered. QNN SDK can be utilized by setting an environment variable or specifying the path directly.<br />There are three ways to input parameters:<br />1. Inputting only the parameter "-path"<br />2. Inputting only the parameter "-m"<br />3. Neither entering the parameter "-m" nor "-path"



### Example

Here, with -path, give your tflite model path i.e. till `*.tflite file`, and it will copy model file to assets folder to build andoid app.
```
    python build_apk.py -q "<QNN_SDK_PATH>" -path "Path/to/TFLITE/Model"
```


Also, you can use AI-HUB Model name as mentioned in models directory, to directly export the model from AI-Hub and copy it to app assets.

```
    python build_apk.py -q "<QNN_SDK_PATH>" -m <Model Name>
```

You can also select the model provided in the list menu during the execution of build_apk.py without specifying the model name and model path.

```
    python build_apk.py -q "<QNN_SDK_PATH>" 
```
