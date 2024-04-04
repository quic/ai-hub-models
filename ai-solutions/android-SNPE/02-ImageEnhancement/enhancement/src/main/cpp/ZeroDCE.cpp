// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
//
// Created by shivmahe on 9/5/2023.
//

#include "ZeroDCE.h"
void ZeroDCE::preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img, std::vector<int> dims)
{
    LOGI("ZeroDCE PREPROCESS is called");
    float * accumulator = reinterpret_cast<float *> (&dest_buffer[0]);
    cv::Mat resized_img;
    cv::resize(img,resized_img,cv::Size(dims[2],dims[1]),cv::INTER_CUBIC);
    LOGI("input image SIZE width%d::%d height%d::%d",dims[1],resized_img.cols, dims[2],resized_img.rows);
    cvtColor(resized_img, resized_img, CV_BGRA2RGB);
    LOGI("num of channels: %d",resized_img.channels());
    int lim = resized_img.rows*resized_img.cols*3;
    for(int idx = 0; idx<lim; idx++){
        float inputScale = 0.00392156862745f;
        accumulator[idx] = resized_img.data[idx] * inputScale;
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