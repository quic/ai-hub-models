// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef __Detection_IMPL_H__
#define __Detection_IMPL_H__

#include <vector>
#include <string>
#include <unistd.h>
#include <memory>

#include "SNPERuntime.h"
#include "ModelInference.h"
#include "Configuration.h"
#include "Detection.h"
namespace detectionsnpe 
{
    class DETECTIONSnpe 
    {
        public:
            DETECTIONSnpe();
            ~DETECTIONSnpe();
            bool Initialize(const ObjectDetectionSnpeConfig& config);
            bool DeInitialize();
            bool Detect(cv::Mat input,cv::Mat& output_image,string model_name);
            bool SetScoreThresh(const float& conf_thresh, const float& nms_thresh);
            bool IsInitialized() const;

        private:
            bool m_isInit; 
            float m_nmsThresh;
            float m_confThresh;
            std::unique_ptr<snperuntime::SNPERuntime> m_snperuntime;
            std::vector<std::string> m_inputLayers;
            std::vector<std::string> m_outputLayers;
            std::vector<std::string> m_outputTensors;

            bool PreProcessInput(const cv::Mat& frame,string model_name);
            bool PostProcess( cv::Mat image,cv::Mat& output_image,string model_name);
            float computeIoU(const cv::Rect& a, const cv::Rect& b);
            std::vector<ObjectData> doNMS(std::vector<ObjectData> winList, const float& nms_thresh);
    };

} // namespace detection

#endif // __DETECTION_IMPL_H__
