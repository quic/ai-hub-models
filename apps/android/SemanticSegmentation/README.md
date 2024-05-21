### Requirements

1. Java, android-sdk and sdkmanager is already set at user's end
2. User should have Linux QNN SDK in local machine.
3. ANDROID_HOME is set to android-sdk path
4. AI-Hub is properly configured with user token.


## Info
Please execute build_apk.py. This script will compile and download a model from AI-Hub and paste it in your Android Project and Generate app-debug.apk
Here, we resize the input image based on model input and gives a segmented map as output which marks every pixek into its corresponding class.

## Preprocessing


```
Bitmap scaledBitmap = Bitmap.createScaledBitmap(modelInputBitmap,inpDims_w, inpDims_h,true);
float[][][][] floatInputArray = new float[1][inpDims_h][inpDims_w][3];
for (int x = 0; x < inpDims_h; x++) {
    for (int y = 0; y < inpDims_w; y++) {
        int pixel = scaledBitmap.getPixel(y, x);
        floatInputArray[0][x][y][0] = (((float)Color.red(pixel))/225f);
        floatInputArray[0][x][y][1] = ((float)Color.green(pixel)/255f);
        floatInputArray[0][x][y][2] = ((float)Color.blue(pixel)/255f);
    }
}

```


## PostProcessing


```
float[] temparr = new float[outDims_w];
for (int i = 0; i < outDims_h; i++) {
    for (int j = 0; j < outDims_w; j++) { // Looping through the columns
        float max_val=Float.NEGATIVE_INFINITY;
        float maxIndex =-1;  // Initializing the max value
        for (int k = 0; k < noOfClasses; k++) { // Looping through the remaining elements in the first axis
            if (output[0][i][j][k] > max_val) { // Comparing the current element with the max value
                maxIndex=k;
                max_val = output[0][i][j][k]; // Updating the max value if the current element is large
            }
        }
        temparr[j]=maxIndex;
    }
    src.put(i,0,temparr);
}

Imgproc.resize(src, dst, dst.size(), 0, 0, Imgproc.INTER_AREA);
int z=0;
for (int x = 0; x <modelInputBitmap.getHeight(); x++) {
    for (int y = 0; y <modelInputBitmap.getWidth(); y++){
        segMap[z++] =(int)dst.get(x, y)[0];
    }
}

```

### Build App:

You have to run build_apk.py for Image Classification. It will generate superresolution-debug.apk and install it in connected device.


    build_apk.py [-h] -q QNNSDK [-m MODEL_NAME] [-path MODEL_PATH]

```
options:

  -h, --help                                         show this help message and exit
  -q QNNSDK, --qnnsdk QNNSDK                         Give path of QNN SDK (REQUIRED)
  -m MODEL_NAME, --model_name MODEL_NAME             Model Name (Optional)
  -path MODEL_PATH, --model_path MODEL_PATH          Model Path (Optional)

```
