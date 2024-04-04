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
// Created by shubgoya on 8/2/2023.
//

#include "QuickSRNetSmall.h"

void QuickSRNetSmall::preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img, std::vector<int> dims)
{
    LOGI("SESR Class Preprocess is called");
    cv::Mat resized_img;

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,resized_img,cv::Size(dims[2],dims[1]),cv::INTER_LINEAR); //Resizing based on input
    LOGI("inputimageSIZE width%d::%d height%d::%d",dims[1],resized_img.cols, dims[2],resized_img.rows);

    float inputScale = 0.00392156862745f;    //normalization value, this is 1/255

    float * accumulator = reinterpret_cast<float *> (&dest_buffer[0]);

    //opencv read in BGRA by default
    cvtColor(resized_img, resized_img, CV_BGRA2RGB);
    LOGI("num of channels: %d",resized_img.channels());
    int lim = resized_img.rows*resized_img.cols*3;
    for(int idx = 0; idx<lim; idx++)
        accumulator[idx]= resized_img.data[idx]*inputScale;

}

void QuickSRNetSmall::postprocess(cv::Mat &outputimg) {
    //This function will multiply by 255 and convert 4byte float value to 1byte int.
    LOGI("SESR Class postprocess");
    outputimg.convertTo(outputimg,CV_8UC3, 255);
}

void QuickSRNetSmall::msg()
{
    LOGI("SESR Class msg");
}