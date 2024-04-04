//
// Created by sbudha on 1/25/2024.
//

#ifndef OBJECTDETECTION_INFERENCE_H
#define OBJECTDETECTION_INFERENCE_H

#include "android/log.h"
#include "Model.h"
#define  LOG_TAG    "QNN_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
std::string build_network(const char * modelPath, const char * backEndPath, char* buffer, long bufferSize);
bool SetAdspLibraryPath(std::string nativeLibPath);
bool executeModel(cv::Mat &img, int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, Model *modelobj);

#endif //OBJECTDETECTION_INFERENCE_H
