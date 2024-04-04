// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include <jni.h>
#include <string>
#include <iostream>

#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <hpp/inference.h>

#include "android/log.h"

#include "hpp/CheckRuntime.hpp"
#include "hpp/SetBuilderOptions.hpp"
#include "hpp/Util.hpp"
#include "LoadContainer.hpp"
#include "CreateUserBuffer.hpp"
#include "LoadInputTensor.hpp"

#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

std::unique_ptr<zdl::SNPE::SNPE> snpe;

std::mutex mtx;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;

zdl::DlSystem::UserBufferMap inputMap, outputMap;
std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationInputBuffers;
int bitWidth = 32;


#include <android/trace.h>
#include <dlfcn.h>
#include <unistd.h>



std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size, const char runtime_arg)
{
    std::string outputLogger;
    bool usingInitCaching = false;

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = nullptr ;

    container = loadContainerFromBuffer(dlc_buffer, dlc_size);

    if (container == nullptr) {
        LOGE("Error while opening the container file.");
        return "Error while opening the container file.\n";
    }

    runtimeList.clear();
    LOGI("runtime arg %c",runtime_arg);
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    if (runtime_arg == 'D'){
        runtime = zdl::DlSystem::Runtime_t::DSP;
        LOGI("Added DSP");
    }
    else if (runtime_arg == 'G')
    {
        runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID; //can be written as GPU
        LOGI("Added GPU");
    }

    if(runtime != zdl::DlSystem::Runtime_t::UNSET)
    {
        bool ret = runtimeList.add(checkRuntime(runtime));
        if(ret == false){
            LOGE("Cannot set runtime");
            return outputLogger + "\nCannot set runtime";
        }
    } else {
        return outputLogger + "\nCannot set runtime";
    }


    mtx.lock();
    snpe= setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);
    mtx.unlock();

    if (snpe== nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for segmentation");
        outputLogger += "Model Prepare failed for segmentation";
        return outputLogger + "SNPE Prepare failed for segmentation";
    }

    outputLogger += "\nsegmentation Model Network Prepare success !!!\n";

    //Creating Buffer
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, useIntBuffer, bitWidth);
    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, useIntBuffer, bitWidth);
    return outputLogger;
}


void executeDLC(cv::Mat &img, int orig_width, int orig_height, float &inferenceTime, cv::Mat &destmat) {

    ATrace_beginSection("preprocessing");

    struct timeval start_time, end_time;
    float seconds, useconds;

    inferenceTime = -1.0f;

    mtx.lock();
    assert(snpe!=nullptr);

    if(!loadInputUserBuffer(applicationInputBuffers, snpe, img, inputMap, bitWidth))
    {
        LOGE("Failed to load Input UserBuffer");
        mtx.unlock();
        return;

    }


    ATrace_endSection();
    gettimeofday(&start_time, NULL);
    ATrace_beginSection("inference time");

    bool execStatus = snpe->execute(inputMap, outputMap);
    ATrace_endSection();

    ATrace_beginSection("postprocessing time");
    gettimeofday(&end_time, NULL);
    seconds = end_time.tv_sec - start_time.tv_sec; //seconds
    useconds = end_time.tv_usec - start_time.tv_usec; //milliseconds
    inferenceTime = ((seconds) * 1000 + useconds/1000.0);

    if(execStatus== true){
        LOGI("Exec segmentation status is true");
    }
    else{
        LOGE("Exec segmentation status is false");
        mtx.unlock();
        return;
    }

    const auto& outputNamesOpt = snpe->getOutputTensorNames();
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    const char* name = outputNames.at(0);

    std::vector<float32_t> segment_seq= applicationOutputBuffers.at(name);


    cv::Mat A (400,400,CV_32F, segment_seq.data());
    cv::resize(A,destmat,cv::Size(orig_width,orig_height),cv::INTER_LINEAR);

    ATrace_endSection();
    mtx.unlock();

    return;
}

