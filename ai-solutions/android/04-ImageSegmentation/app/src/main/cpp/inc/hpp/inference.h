// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================

#ifndef NATIVEINFERENCE_INFERENCE_H
#define NATIVEINFERENCE_INFERENCE_H

#include "zdl/DlSystem/TensorShape.hpp"
#include "zdl/DlSystem/TensorMap.hpp"
#include "zdl/DlSystem/TensorShapeMap.hpp"
#include "zdl/DlSystem/IUserBufferFactory.hpp"
#include "zdl/DlSystem/IUserBuffer.hpp"
#include "zdl/DlSystem/UserBufferMap.hpp"
#include "zdl/DlSystem/IBufferAttributes.hpp"

#include "zdl/DlSystem/StringList.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlVersion.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"

#include "zdl/DlSystem/ITensor.hpp"
#include "zdl/DlSystem/ITensorFactory.hpp"

#include <unordered_map>
#include "android/log.h"
#include <opencv2/opencv.hpp>

#define  LOG_TAG    "SNPE_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

class BoxCornerEncoding {

public:
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    std::string objlabel;

    BoxCornerEncoding(int a, int b, int c, int d,int sc, std::string name="person")
    {
        x1 = a;
        y1 = b;
        x2 = c;
        y2 = d;
        score = sc;
        objlabel = name;
    }
};

std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size, const char runtime_arg);
bool SetAdspLibraryPath(std::string nativeLibPath);

void executeDLC(cv::Mat &img, int orig_width, int orig_height, float &inferenceTime, cv::Mat &dest);

#endif //NATIVEINFERENCE_INFERENCE_H
