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
// Created by shubpate on 12/11/2021.
//

#ifndef NATIVEINFERENCE_INFERENCE_H
#define NATIVEINFERENCE_INFERENCE_H

std::string build_network(const uint8_t * dlc_buffer, const size_t dlc_size);
bool SetAdspLibraryPath(std::string nativeLibPath);
std::string execute_net(std::vector<float *> inputVec, int arrayLength,
                        std::vector<float *> & outputVec, std::string runtime);

#include "zdl/DlSystem/TensorShape.hpp"
#include "zdl/DlSystem/TensorMap.hpp"
#include "zdl/DlSystem/TensorShapeMap.hpp"
#include "zdl/DlSystem/IUserBufferFactory.hpp"
#include "zdl/DlSystem/IUserBuffer.hpp"
#include "zdl/DlSystem/UserBufferMap.hpp"
#include "zdl/DlSystem/IBufferAttributes.hpp"

#include "zdl/DlSystem/StringList.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlVersion.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"

#include "zdl/DlSystem/ITensor.hpp"
#include "zdl/DlSystem/ITensorFactory.hpp"

#include <unordered_map>
#include "android/log.h"

#define  LOG_TAG    "SNPE_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

bool loadInputUserBuffer(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                            std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                            std::vector<float *> inVector,
                            int arrayLength,
                            zdl::DlSystem::UserBufferMap& inputMap,
                            int bitWidth);

bool saveOutput (zdl::DlSystem::UserBufferMap& outputMap,
                 std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                 std::vector<float *> & outputVec,
                 size_t batchSize,
                 bool isTfNBuffer,
                 int bitWidth);

#endif //NATIVEINFERENCE_INFERENCE_H
