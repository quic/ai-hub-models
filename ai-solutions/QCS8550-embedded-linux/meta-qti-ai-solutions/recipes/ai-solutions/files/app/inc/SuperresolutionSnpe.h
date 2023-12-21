// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef __SUPERRES_IMPL_H__
#define __SUPERRES_IMPL_H__

#include <vector>
#include <string>
#include <unistd.h>
#include <memory>

#include "SNPERuntime.h"
#include "ModelInference.h"
#include "Configuration.h"

namespace superressnpe 
{
    class SUPERRESSnpe 
    {
        public:
            SUPERRESSnpe();
            ~SUPERRESSnpe();
            bool Initialize(const ObjectDetectionSnpeConfig& config);
            bool DeInitialize();
            bool Detect(cv::Mat input,cv::Mat& output_image,string model_name);
            bool IsInitialized() const;

        private:
            bool m_isInit;
            std::unique_ptr<snperuntime::SNPERuntime> m_snperuntime;
            std::vector<std::string> m_inputLayers;
            std::vector<std::string> m_outputLayers;
            std::vector<std::string> m_outputTensors;

            bool PreProcessInput(const cv::Mat& frame,string model_name);
            bool PostProcess(cv::Mat& output_image,string model_name);
    };
} // namespace superressnpe

#endif // __SUPERRES_IMPL_H__
