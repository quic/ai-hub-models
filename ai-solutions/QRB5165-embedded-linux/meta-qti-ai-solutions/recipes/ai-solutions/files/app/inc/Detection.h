// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef DETECTION_H
#define DETECTION_H

#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <memory>

using namespace std;
using namespace cv;

struct ObjectData {
    // Bounding box information: top-left coordinate and width, height
    cv::Rect bbox;
    // Confidence of this bounding box
    float confidence = -1.0f;
    // The label of this Bounding box
    int label = -1;
    // Time cost of detecting this frame
    size_t time_cost = 0;
    uint32_t Width=512;
    uint32_t Height=512;
    cv::Mat *output=NULL;
    
};

struct Detection
{
    cv::Rect bbox;
    float score;
    int label;
};

struct DetectionDetail
{
    vector<Detection> Result;
    string ModelName;
};

struct DetectionItem
{
    uint32_t Width;
    uint32_t Height;
    uint32_t FrameId;
    size_t Size; 
    string StreamName;
    int StreamId;
    shared_ptr<uint8_t> ImageBuffer;   
//    vector<DetectionDetail> Results;
    ObjectData Results;
};

#endif
