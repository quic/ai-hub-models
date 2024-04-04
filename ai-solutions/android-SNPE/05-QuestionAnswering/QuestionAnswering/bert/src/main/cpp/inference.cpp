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
#include "hpp/LoadContainer.hpp"
#include "hpp/SetBuilderOptions.hpp"
#include "hpp/LoadInputTensor.hpp"
#include "hpp/CreateUserBuffer.hpp"
#include "hpp/Util.hpp"

std::unique_ptr<zdl::SNPE::SNPE> snpe_dsp;
std::unique_ptr<zdl::SNPE::SNPE> snpe_cpu;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;

std::string dlc_path;

std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size)
{
    std::string outputLogger;
    bool usingInitCaching = true;

    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = loadContainerFromBuffer(dlc_buffer, dlc_size);
    if (container == nullptr) {
        LOGE("Error while opening the container file.");
        return "Error while opening the container file.\n";
    }

    runtimeList.clear();
    // Build for DSP runtime
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP;
    if(runtime != zdl::DlSystem::Runtime_t::UNSET)
    {
        bool ret = runtimeList.add(checkRuntime(runtime));
            if(ret == false){
                LOGE("Cannot set runtime");
                return outputLogger + "Cannot set runtime";
            }
    } else return outputLogger + "\nCannot set runtime";

    snpe_dsp = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);

    if (snpe_dsp == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for DSP runtime");
        outputLogger += "DSP Build = Failed ; ";
//        return outputLogger + "SNPE Prepare failed for DSP runtime";
    }
    outputLogger += "DSP Build = OK ; ";
    if (usingInitCaching) {
        if (container->save(dlc_path)) {
            LOGI("Saved container into archive successfully");
//            outputLogger += "\nSaved container cache";
        } else LOGE("Failed to save container into archive");
    }

    // Build for CPU runtime
    runtimeList.clear();
    runtime = zdl::DlSystem::Runtime_t::CPU;
    snpe_cpu = setBuilderOptions(container, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);
    if (snpe_cpu == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for CPU runtime");
        return outputLogger += "CPU Build = Failed ; ";
    }

    outputLogger += "CPU Build = OK ; ";
    return outputLogger;
}

// input vector, runtime
std::string execute_net(std::vector<float *> inputVec, int arrayLength,
                        std::vector<float *> & outputVec, std::string runtime) {
    bool execStatus;
    std::unique_ptr<zdl::SNPE::SNPE> snpe;

    // Transfer object properties
    if (runtime =="CPU") {
        snpe = std::move(snpe_cpu);
        LOGI("Executing on CPU runtime...");
    } else snpe = std::move(snpe_dsp);

    // do some exception checking
    if (snpe == nullptr)
        return "SNPE " + runtime + " is NULL ptr";

    zdl::DlSystem::UserBufferMap inputMap, outputMap;
    std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
    std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

    // create UB_TF_N type buffer, if : useIntBuffer=True
    int bitWidth = 32;
    if(useIntBuffer)
        bitWidth = 8;  // set 16 for INT_16 activations

    LOGI("Using UserBuffer with bit-width = %d", bitWidth);

    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, useIntBuffer, bitWidth);

    std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, useIntBuffer, bitWidth);

    if(!loadInputUserBuffer(applicationInputBuffers, snpe, inputVec, arrayLength, inputMap, bitWidth))
        return "\nFailed to load Input UserBuffer";

    // Execute the input buffer map on the model with SNPE
    execStatus = snpe->execute(inputMap, outputMap);
    // Save the execution results only if successful
    if (execStatus == true) {
        LOGI("SNPE Exec Success !!!");
        // save output tensor
        size_t batchSize = 1;
        if(!saveOutput(outputMap, applicationOutputBuffers, outputVec, batchSize, useIntBuffer, bitWidth))
            return "\nFailed to Save Output Tensor";

    } else return "\nSNPE Execute Failed\n";

    // Transfer object properties
    if (runtime == "CPU") {
        snpe_cpu = std::move(snpe);
        LOGI("Transferred back object to CPU runtime...");
    } else snpe_dsp = std::move(snpe);

//    snpe.reset();
    return "";
}
