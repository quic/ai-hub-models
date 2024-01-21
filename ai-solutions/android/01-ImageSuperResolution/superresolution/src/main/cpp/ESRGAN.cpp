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

#include "ESRGAN.h"

void ESRGAN::preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img, std::vector<int> dims)
{
    LOGI("ESRGAN_PREPROCESS is called");
    cv::Mat resized_img;

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,resized_img,cv::Size(dims[2],dims[1]),cv::INTER_LINEAR); //Not needed for this case
    LOGI("inputimageSIZE width%d::%d height%d::%d",dims[1],resized_img.cols, dims[2],resized_img.rows);

    float * accumulator = reinterpret_cast<float *> (&dest_buffer[0]);

    //opencv read in BGRA by default, converting to BGR
    cvtColor(resized_img, resized_img, CV_BGRA2RGB);
    LOGI("num of channels: %d",resized_img.channels());
    int lim = resized_img.rows*resized_img.cols*3;
    for(int idx = 0; idx<lim; idx++)
        accumulator[idx]= resized_img.data[idx];

}

void ESRGAN::postprocess(cv::Mat &outputimg){
    LOGI("ESRGAN Class postprocess");
    //This function will multiply by 1 and convert 4byte float value to 1byte int.
    outputimg.convertTo(outputimg,CV_8UC3, 1);
}

void ESRGAN::msg()
{
    LOGI("ESRGAN Class msg");
}