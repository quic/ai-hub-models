//
// Created by shubgoya on 8/2/2023.
//



#ifndef IMAGEENHANCEMENT_MODEL_H
#define IMAGEENHANCEMENT_MODEL_H

#include <jni.h>
#include <string>
#include <iostream>

#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "android/log.h"


#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#define  LOG_TAG    "QNN_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)


class Model {

public:
    virtual void preprocess(cv::Mat &img, std::vector<int> dims) = 0;
    virtual void postprocess(cv::Mat &outputimg) = 0;
    virtual void msg() = 0;

    std::string model_name="RUAS"; //initialized

};


#endif //IMAGEENHANCEMENT_MODEL_H
