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

std::unique_ptr<zdl::SNPE::SNPE> snpe_HRNET;
std::unique_ptr<zdl::SNPE::SNPE> snpe_BB;

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

std::string build_network_BB(const uint8_t * dlc_buffer_BB, const size_t dlc_size_BB, const char runtime_arg, ModelName modelName)
{
    std::string outputLogger;
    bool usingInitCaching = false;  //shubham: TODO check with true

    std::unique_ptr<zdl::DlContainer::IDlContainer> container_BB = nullptr ;

    container_BB = loadContainerFromBuffer(dlc_buffer_BB, dlc_size_BB);

    if (container_BB == nullptr) {
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
    snpe_BB = setBuilderOptions(container_BB, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching, modelName);
    mtx.unlock();

    if (snpe_BB == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for BB");
        outputLogger += "Model Prepare failed for BB";
        return outputLogger + "SNPE Prepare failed for BB";
    }

    outputLogger += "\nBB Model Network Prepare success !!!\n";

    //Creating Buffer
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe_BB, useIntBuffer, bitWidth);
    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe_BB, useIntBuffer, bitWidth);
    return outputLogger;
}



bool executeDLC(cv::Mat &img, int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, Model *modelobj) {

    LOGI("execute_net_BB");
    ATrace_beginSection("preprocessing");

    struct timeval start_time, end_time;
    float milli_time, seconds, useconds;

    mtx.lock();
    assert(snpe_BB!=nullptr);

    if(!loadInputUserBuffer(applicationInputBuffers, snpe_BB, img, inputMap, bitWidth, modelobj))
    {
        LOGE("Failed to load Input UserBuffer");
        mtx.unlock();
        return false;
    }

    //std::string name_out_boxes = "885";
    //std::string name_out_classes =  "877";

    // get output tensor names of the network that need to be populated
    const auto &outputNamesOpt = snpe_BB->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList &outputNames = *outputNamesOpt;
    assert(outputNames.size() > 0);

    if (outputNames.size()) LOGI("Preprocessing and loading in application Output Buffer for BB");

    std::string name_out_boxes;

    //YoloX is using only single output tensor
    if (modelobj->model_name != YoloX) {
        name_out_boxes = outputNames.at(1);
        LOGI("Filling %s buffer name_out_boxes", name_out_boxes.c_str());
    }


    std::string name_out_classes =  outputNames.at(0);
    LOGI("Filling %s buffer name_out_classes", name_out_classes.c_str());

    ATrace_endSection();
    gettimeofday(&start_time, NULL);
    ATrace_beginSection("inference time");

    bool execStatus = snpe_BB->execute(inputMap, outputMap);
    ATrace_endSection();
    ATrace_beginSection("postprocessing time");
    gettimeofday(&end_time, NULL);
    seconds = end_time.tv_sec - start_time.tv_sec; //seconds
    useconds = end_time.tv_usec - start_time.tv_usec; //milliseconds
    milli_time = ((seconds) * 1000 + useconds/1000.0);
    //LOGI("Inference time %f ms", milli_time);

    if(execStatus== true){
        LOGI("Exec BB status is true");
    }
    else{
        LOGE("Exec BB status is false");
        mtx.unlock();
        return false;
    }

    std::vector<float32_t> BBout_boxcoords;

    //YoloX is using only single output tensor
    if (modelobj->model_name != YoloX) {
        LOGI("reading output name_out_boxes");
        BBout_boxcoords = applicationOutputBuffers.at(name_out_boxes);
    }

    LOGI("reading output name_out_classes");
    std::vector<float32_t> BBout_class = applicationOutputBuffers.at(name_out_classes);
    //LOGI("reading output done. Calling postprocess");

    modelobj->postprocess(orig_width, orig_height, numberofobj, BB_coords, BB_names, BBout_boxcoords, BBout_class, milli_time);

    ATrace_endSection();
    mtx.unlock();
    return true;
}

