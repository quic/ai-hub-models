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

#include <thread>
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

std::unique_ptr<zdl::SNPE::SNPE> snpe;

std::mutex mtx;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;

bool execStatus_thread = false;
zdl::DlSystem::UserBufferMap inputMap, outputMap;
std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationInputBuffers;
int bitWidth = 32;


#include <android/trace.h>
#include <dlfcn.h>
#include <opencv2/gapi/core.hpp>

std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size, const char runtime_arg)
{
    std::string outputLogger;
    bool usingInitCaching = false;  //shubham: TODO check with true

    std::unique_ptr<zdl::DlContainer::IDlContainer> container_snpe = nullptr ;

    container_snpe = loadContainerFromBuffer(dlc_buffer, dlc_size);

    if (container_snpe == nullptr) {
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
    snpe = setBuilderOptions(container_snpe, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);
    mtx.unlock();

    if (snpe == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed");
        outputLogger += "Model Prepare failed";
        return outputLogger + "SNPE Prepare failed";
    }

    outputLogger += "\nModel Network Prepare success !!!\n";

    //Creating Buffer
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, useIntBuffer, bitWidth);
    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, useIntBuffer, bitWidth);
    return outputLogger;
}

void executeonthread()
{
    LOGI("shubham thread is running");
    if(snpe== nullptr)
        LOGE("SNPE IS NULL");
    execStatus_thread  = snpe->execute(inputMap, outputMap);
}

bool executeDLC(cv::Mat &inputimg, cv::Mat &outputimg, float &milli_time, Model *modelobj) {

    LOGI("execute_DLC");
    ATrace_beginSection("preprocessing");

    struct timeval start_time, end_time;
    float seconds, useconds;

    mtx.lock();
    assert(snpe != nullptr);

    if(!loadInputUserBuffer(applicationInputBuffers, snpe, inputimg, inputMap, bitWidth, modelobj))
    {
        LOGE("Failed to load Input UserBuffer");
        mtx.unlock();
        return false;
    }

    ATrace_endSection();
    gettimeofday(&start_time, NULL);
    ATrace_beginSection("inference time");

    std::thread t1(executeonthread);
    LOGI("shubham waiting");
    t1.join();
    bool execStatus = execStatus_thread;
//    bool execStatus = snpe->execute(inputMap, outputMap);
    ATrace_endSection();
    ATrace_beginSection("postprocessing time");
    gettimeofday(&end_time, NULL);
    seconds = end_time.tv_sec - start_time.tv_sec; //seconds
    useconds = end_time.tv_usec - start_time.tv_usec; //milliseconds
    milli_time = ((seconds) * 1000 + useconds/1000.0);
    //LOGI("Inference time %f ms", milli_time);

    if(execStatus== true){
        LOGI("Exec status is true");
    }
    else{
        LOGE("Exec status is false");
        mtx.unlock();
        return false;
    }

    const auto& outputNamesOpt = snpe->getOutputTensorNames();
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    const char* name = outputNames.at(0);

    LOGI("outbut buffers: %s", name);
    std::vector<float32_t> databuffer = applicationOutputBuffers.at(name);
    std::vector<int> dims;
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);

    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
    int num_dims = bufferShape.rank();
    for(int i=0;i<num_dims;i++)
    {
        LOGI("dims[%d]: %d, ",i,bufferShape[i]);
        dims.push_back(bufferShape[i]);
    }


    outputimg = cv::Mat(bufferShape[1], bufferShape[2], CV_32FC3, databuffer.data(),
                        cv::Mat::AUTO_STEP);
    modelobj->postprocess(outputimg);

    ATrace_endSection();
    mtx.unlock();
    return true;
}

