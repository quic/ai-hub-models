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
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <fstream>
#include <unistd.h>
#include "android/log.h"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlVersion.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"
#include "zdl/DlSystem/ITensor.hpp"
#include "zdl/DlSystem/StringList.hpp"
#include "zdl/DlSystem/TensorMap.hpp"
#include "zdl/DlSystem/TensorShape.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "hpp/LoadInputTensor.hpp"
#include "hpp/Util.hpp"
#include "inference.h"

bool SetAdspLibraryPath(std::string nativeLibPath) {
    nativeLibPath += ";/data/local/tmp/mv_dlc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";

    __android_log_print(ANDROID_LOG_INFO, "SNPE ", "ADSP Lib Path = %s \n", nativeLibPath.c_str());
    std::cout << "ADSP Lib Path = " << nativeLibPath << std::endl;

    return setenv("ADSP_LIBRARY_PATH", nativeLibPath.c_str(), 1 /*override*/) == 0;
}


std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromBuffer(const uint8_t * buffer, const size_t size)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(buffer, size);
    return container;
}


zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    LOGI("SNPE Version = %s", Version.asString().c_str()); //Print Version number

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        LOGE("Selected runtime not present. Falling back to GPU.");
        runtime = zdl::DlSystem::Runtime_t::GPU;
        if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)){
            LOGE("Selected runtime not present. Falling back to CPU.");
            runtime = zdl::DlSystem::Runtime_t::CPU;
        }
    }

    return runtime;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   bool useUserSuppliedBuffers,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    std::string platformOptionStr = "useAdaptivePD:ON";

    zdl::DlSystem::PlatformConfig platformConfig;
    bool setSuccess = platformConfig.setPlatformOptions(platformOptionStr);
    if (!setSuccess)
        LOGE("=========> failed to set platformconfig: %s", platformOptionStr.c_str());
    else
        LOGI("=========> platformconfig set: %s", platformOptionStr.c_str());

    bool isValid = platformConfig.isOptionsValid();
    if (!isValid)
        LOGE("=========> platformconfig option is invalid");
    else
        LOGI("=========> platformconfig option: valid");


    zdl::DlSystem::StringList stringruntime = runtimeList.getRuntimeListNames();
    for (const char *name : stringruntime)
        LOGI("runtime sh %s", name);

    snpe = snpeBuilder.setOutputLayers({})
            .setExecutionPriorityHint(
                    zdl::DlSystem::ExecutionPriorityHint_t::HIGH)
            .setRuntimeProcessorOrder(runtimeList)
            .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
            .setPlatformConfig(platformConfig)
            .setInitCacheMode(useCaching)
            .build();

    return snpe;
}

// ==============================User Buffer func=================================== //
// ================================================================================= //


//CreateUserbuffer INPUT/OUTPUT for segmentation
void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, std::vector<float32_t>>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name,
                      const bool isTfNBuffer,
                      int bitWidth)
{

    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);

    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();

    size_t bufferElementSize = 0;
    if (isTfNBuffer) {
        bufferElementSize = bitWidth / 8;
    }
    else {
        bufferElementSize = sizeof(float);
    }

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.

    int num_dims = bufferShape.rank(); //bufferShape rank is generally 1 more than expected, as it add 1 for batchSize, so 320x320x3 will look like 1x320x320x3
    std::vector<size_t> strides(num_dims);
    strides[strides.size() - 1] = bufferElementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = num_dims - 1; i > 0; i--) {
        stride *= bufferShape[i];
        strides[i - 1] = stride;
    }

    size_t bufSize=bufferElementSize;
    for(int i=0;i<num_dims;i++)
        bufSize*=bufferShape[i];

    LOGI("\n BufferName: %s Bufsize: %d", name, bufSize);

    // set the buffer encoding type
    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    if (isTfNBuffer)
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(
                new zdl::DlSystem::UserBufferEncodingTfN(0, 1.0, bitWidth));
    else
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(
                new zdl::DlSystem::UserBufferEncodingFloat());

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<float32_t>(bufSize));

    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory &ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(
            ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                       bufSize,
                                       strides,
                                       userBufferEncoding.get()));
    if (snpeUserBackedBuffers.back() == nullptr)
        throw std::runtime_error(std::string("Error while creating user buffer."));

    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());

}

/*
 Cretae OutPut Buffer Map for segmentation
 */
void createOutputBufferMap(zdl::DlSystem::UserBufferMap& outputMap,
                           std::unordered_map<std::string, std::vector<float32_t>>& applicationBuffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool isTfNBuffer,
                           int bitWidth)
{
    const auto& outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char *name : outputNames) {
        LOGI("Creating output buffer %s", name);
        createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}
/*
 * Create Input Buffer Map for segmentation
 */
void createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                          std::unordered_map<std::string, std::vector<float32_t>>& applicationBuffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool isTfNBuffer,
                          int bitWidth) {
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList &inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name: inputNames) {
        LOGI("Creating Input Buffer = %s", name);
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name,
                         isTfNBuffer, bitWidth);
    }
}

void preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img) {
    cv::Mat img_rez;
    cv::resize(img, img_rez, cv::Size(400, 400),
               cv::INTER_LINEAR);  //TODO get the size from model itself

    float inputScale = 0.00784313771874f;    //normalization value, this is 1/255

    float *accumulator = reinterpret_cast<float *> (&dest_buffer[0]);

    /*opencv read in BGRA by default
     * but when we converted Bitmap to Mat in Java
     * Blue and Red channels got swapped,
     * so after doing below, mat will be in actual BGR.
     */
    cvtColor(img_rez, img_rez, CV_BGRA2RGB);
    LOGI("num of channels: %d", img_rez.channels());
    int lim = img_rez.rows * img_rez.cols * 3;
    for (int idx = 0; idx < lim; idx++)
    {
        accumulator[idx] = img_rez.data[idx] * inputScale - 1.0f;
    }

}


//Preprocessing and loading in application Input Buffer for segmentation
bool loadInputUserBuffer(std::unordered_map<std::string, std::vector<float32_t>>& applicationBuffers,
                            std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                            cv::Mat &img,
                            zdl::DlSystem::UserBufferMap& inputMap,
                            int bitWidth) {

    // get input tensor names of the network that need to be populated
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList &inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) LOGI("Preprocessing and loading in application Input Buffer for segmentation");


    for (size_t j = 0; j < inputNames.size(); j++) {
        const char *name = inputNames.at(j);
        LOGI("Filling %s buffer ", name);

        if(bitWidth == 8 || bitWidth == 16) {
            LOGE("bitwidth 8 and 16 are NOT DEFINED");
            return false;
        } else {

            preprocess(applicationBuffers.at(name),img);  //functions loads data in applicationBuffer

        }
    }
    return true;
}
