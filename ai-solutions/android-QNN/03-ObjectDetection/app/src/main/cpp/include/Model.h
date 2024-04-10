//
// Created by shubgoya on 8/2/2023.
//

#ifndef APP_MODEL_H
#define APP_MODEL_H

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

// List of All the supported models by the current application
enum ModelName
{
    YOLONAS,
    SSDMobilenetV2,
    YoloX
};


class Model {
public:
    virtual void preprocess(cv::Mat &img, std::vector<int> dims) = 0;
    virtual void postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time) = 0;
    virtual void msg() = 0;

    ModelName model_name=YOLONAS; //initialized

};


#endif //APP_MODEL_H

