// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "Configuration.h"
#include "SuperresolutionSnpe.h"

namespace superressnpe {

    /** @brief Constructor
    */
    SUPERRESSnpe::SUPERRESSnpe() : m_isInit(false),m_snperuntime(nullptr)
    {

    }

    /** @brief Destructor
    */
    SUPERRESSnpe::~SUPERRESSnpe() {
        DeInitialize();
    }

    /** @brief To read model config and set output layers
     * @param config model config parameters
     * @return true if success;false otherwise
    */
    bool SUPERRESSnpe::Initialize(const ObjectDetectionSnpeConfig& config)
    {
        m_snperuntime = std::move(std::unique_ptr<snperuntime::SNPERuntime>(new snperuntime::SNPERuntime()));

        m_inputLayers = config.inputLayers;
        m_outputLayers = config.outputLayers;
        m_outputTensors = config.outputTensors;

        /**
         * To set output layer from model config
        */
        m_snperuntime->SetOutputLayers(m_outputLayers);
        /**
         * To initialize snperuntime 
        */
        if (!m_snperuntime->Initialize(config.model_path, config.runtime)) {
            LOG_ERROR("Failed to Initialize snpe instance.\n");
            return false;
        }
        m_isInit = true;
        return true;
    }

    /** @brief To deallocate buffers
    */
    bool SUPERRESSnpe::DeInitialize()
    {
        if (m_isInit) {
            m_snperuntime->Deinitialize();
            m_snperuntime.reset(nullptr);
        }
        m_isInit = false;
        return true;
    }

    bool SUPERRESSnpe::IsInitialized() const
    {
            return m_isInit;
    }

    /** @brief To preprocess input image
     * @param input_image Input image for inference
     * @return true if succuess; false otherwise
    */
    bool SUPERRESSnpe::PreProcessInput(const cv::Mat& input_image,string model_name)
    {
        if (input_image.empty()) 
        {
            LOG_ERROR("Invalid image!\n");
            return false;
        }

        auto inputShape = m_snperuntime->GetInputShape(m_inputLayers[0]);
        size_t model_h = inputShape[1];
        size_t model_w = inputShape[2];
        
        if (m_snperuntime->GetInputTensor(m_inputLayers[0]) == nullptr) 
        {
            LOG_ERROR("Empty input tensor\n");
            return false;
        }

        cv::Mat image;
        cv::resize(input_image,image,cv::Size(model_h,model_w),cv::INTER_CUBIC);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::Mat input(model_h, model_w, CV_32FC3,  m_snperuntime->GetInputTensor(m_inputLayers[0]));
        if(model_name.compare("ESRGAN") == 0)
        {
            image.convertTo(input, CV_32FC3, 1.0);
        }
        else
        {
            image.convertTo(input, CV_32FC3, 1.0/255.0);
        }
        return true;
    }

    /** @brief To preprocess,execute and postprocess
     * @param input_image Input image for inference
     * @param output_image Inference output image
     * @param model_name To identify model for specific post-processing
     * @return true if success; false otherwise
    */
    bool SUPERRESSnpe::Detect(cv::Mat input_image,cv::Mat& output_image, string model_name)
    {
        /**
         * Preprocessing image
        */
        if(PreProcessInput(input_image, model_name) != true)
        {
            LOG_ERROR("PreProcess failed\n");
            return false;
        }
        /**
         * Inferencing model on target
        */
        if (!m_snperuntime->execute()) {
            LOG_ERROR("SNPERuntime execute failed.");
            return false;
        }
        /**
         * Postprocessing
        */
        if(PostProcess(output_image, model_name) != true)
        {
            LOG_ERROR("PostProcess failed\n");
            return false;
        }
        return true;
    }

    /** @brief Superres postprocess 
     * @param output_image upscaled image
     * @param model_name To identify model for specific post-processing
    */
    bool SUPERRESSnpe::PostProcess(cv::Mat& output_image,string model_name)
    {
        auto outputShape = m_snperuntime->GetOutputShape(m_outputTensors[0]);
        float *output =  m_snperuntime->GetOutputTensor(m_outputTensors[0]);

        if(output == nullptr)
        {
            return false;
        }
        int height = outputShape[1];
        int width = outputShape[2];

        output_image = cv::Mat(cv::Size(width,height), CV_32FC3, output);
        if(model_name.compare("ESRGAN") == 0)
        {
            output_image.convertTo(output_image, CV_8UC3, 1.0);
        }
        else
        {
            output_image.convertTo(output_image, CV_8UC3, 255.0);
        }
        cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
        return true;
    }

} // namespace superressnpe
