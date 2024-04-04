//
// Created by sbudha on 1/25/2024.
//

#ifndef IMAGESEGMENTATION_INFERENCE_H
#define IMAGESEGMENTATION_INFERENCE_H

#include "android/log.h"

#define  LOG_TAG    "QNN_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
std::string build_network(const char * modelPath, const char * backEndPath, char* buffer, long bufferSize);
bool SetAdspLibraryPath(std::string nativeLibPath);
bool executeModel(cv::Mat &img, int orig_width, int orig_height, float &milli_time, cv::Mat &destmat);

#endif //IMAGESEGMENTATION_INFERENCE_H
