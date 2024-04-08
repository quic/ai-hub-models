//
// Created by shivmahe on 9/5/2023.
//

#include "../include/ZeroDCE.h"
#include <opencv2/imgcodecs.hpp>

void ZeroDCE::preprocess(cv::Mat &img, std::vector<int> dims)
{
    LOGI("ZeroDCE PREPROCESS is called");

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]),cv::INTER_CUBIC); //Resizing based on input
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

void ZeroDCE::postprocess(cv::Mat &outputimg){
    LOGI("ZeroDCE Class post-process");
    outputimg.convertTo(outputimg,CV_8UC3, 255);
}

void ZeroDCE::msg()
{
    LOGI("ZeroDCE Class msg");
}