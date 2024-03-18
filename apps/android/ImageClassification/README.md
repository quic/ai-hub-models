### Requirements

1. Java, android-sdk and sdkmanager is already set at user's end
2. User should have Linux QNN SDK in local machine.


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


    build_apk.py [-h] -q QNNSDK (-m MODEL_PATH | -e MODEL_NAME)



### Example

Here, with -m, give your tflite model path i.e. till `*.tflite file`, and it will copy model file to assets folder to build andoid app.
```
    python build_apk.py -q "<QNN_SDK_PATH>" -m "Path\to\TFLITE\Model"
```

Also, you can use AI-HUB Model name as mentioned in models directory, to directly export the model from AI-Hub and copy it to app assets.

```
    python build_apk.py -q "<QNN_SDK_PATH>" -e <Model Name>
```
