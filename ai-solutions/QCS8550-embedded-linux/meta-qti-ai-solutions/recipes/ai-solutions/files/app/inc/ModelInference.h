// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef MODEL_INFERENCE_H_
#define MODEL_INFERENCE_H_
#include <vector>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "Configuration.h"

class ModelInference{
public:
    ModelInference();
    ModelInference(const string model_name);
    int Initialization(const ObjectDetectionSnpeConfig& config);
    bool IsInitialized();
    bool UnInitialization();
    ~ModelInference();
    int Inference(cv::Mat input,cv::Mat& output_image,string model_name);
private:
    void *Impl  = nullptr;
    enum Models{SUPERRESOLUTION, DETECTION,LOWLIGHT,SEGMENTATION};
    int Model;
};

#endif