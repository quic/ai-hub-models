//
// Created by shubgoya on 8/2/2023.
//

#include <opencv2/imgcodecs.hpp>
#include "../include/SRGAN.h"

void SRGAN::preprocess(cv::Mat &img, std::vector<int> dims)
{
    LOGI("SRGAN Class Preprocess is called");

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]),cv::INTER_LINEAR); //Resizing based on input
    LOGI("inputimage SIZE width::%d height::%d",img.cols, img.rows);

    float inputScale = 0.00392156862745f;    //normalization value, this is 1/255

    //opencv read in BGRA by default
    cvtColor(img, img, CV_BGRA2RGB);
    img.convertTo(img,CV_32FC3,inputScale);
    LOGI("num of channels: %d",img.channels());

    float* temp = reinterpret_cast<float *>(img.data);
    for(int i=0;i<10;i++)
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "input buffer[%d] = %f\n",i,temp[i]);
    }
}

void SRGAN::postprocess(cv::Mat &outputimg) {
    //This function will multiply by 255 and convert 4byte float value to 1byte int.
    LOGI("postprocess");
    LOGI("width = %d",outputimg.cols);
    LOGI("height = %d\n",outputimg.rows);
    LOGI("channel = %d\n",outputimg.channels());
    outputimg.convertTo(outputimg,CV_8UC3, 255);
    LOGI("postprocess done");
}

void SRGAN::msg()
{
    LOGI("SRGAN Class msg");
}