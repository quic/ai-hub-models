### Requirements

1. Java, android-sdk and sdkmanager is already set at user's end
2. User should have Linux QNN SDK in local machine.
3. ANDROID_HOME is set to android-sdk path
4. AI-Hub is properly configured with user token.


## Info
Please execute build_apk.py. This script will compile and download a model from AI-Hub and paste it in your Android Proect and Generate superresolution-debug.apk

This app takes model with image of size 128x128 as input and gives 512x512 as output. If you want, you can replace the model with any superesolution tflite model, but you have to change the pre-processing, post-processing and dimensions in the app code based on model parameters.


## Preprocessing


```
    public void PreProcess(Bitmap inputBitmap, int input_dims1, int input_dims2, float[][][][] floatinputarray){
        for (int x = 0; x < input_dims1; x++) {
            for (int y = 0; y < input_dims2; y++) {
                int pixel = inputBitmap.getPixel(x, y);
                // Normalize channel values to [-1.0, 1.0]. Here, pixel values
                // are positive so the effective range will be [0.0, 1.0]
                floatinputarray[0][x][y][0] = (Color.red(pixel))/255.0f;
                floatinputarray[0][x][y][1] = (Color.green(pixel))/255.0f;
                floatinputarray[0][x][y][2] = (Color.blue(pixel))/255.0f;
            }
        }
    }
```


## PostProcessing


```
    public void PostProcess(Bitmap outbmp, int output_dims1, int output_dims2, float[][][][] floatoutputarray) {
        for (int x = 0; x < output_dims1; x++) {
            for (int y = 0; y < output_dims2; y++) {
                int red = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][0] * 255)));
                int green = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][1] * 255)));
                int blue = (int) (Math.max(0, Math.min(255, floatoutputarray[0][x][y][2] * 255)));
                int color = Color.argb(255, red, green, blue);
                outbmp.setPixel(x, y, color);
            }
        }
    }
```

### Build App:

You have to run build_apk.py for Image Classification. It will generate classification-debug.apk and install it in connected device.


    build_apk.py [-h] -q QNNSDK [-m MODEL_NAME] [-path MODEL_PATH]

```
options:

  -h, --help                                         show this help message and exit
  -q QNNSDK, --qnnsdk QNNSDK                         Give path of QNN SDK (REQUIRED)
  -m MODEL_NAME, --model_name MODEL_NAME             Model Name (Optional)
  -path MODEL_PATH, --model_path MODEL_PATH          Model Path (Optional)

```
