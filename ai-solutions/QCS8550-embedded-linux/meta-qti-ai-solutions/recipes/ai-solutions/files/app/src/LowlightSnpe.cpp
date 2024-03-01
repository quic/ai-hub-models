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
#include "LowlightSnpe.h"

namespace lowlightsnpe 
{

    /** @brief Constructor
    */
    LOWLIGHTSnpe::LOWLIGHTSnpe() :  m_isInit(false),m_snperuntime(nullptr) 
    {

    }

    /** @brief Destructor
    */
    LOWLIGHTSnpe::~LOWLIGHTSnpe() 
    {
        DeInitialize();
    }

    /** @brief To read model config and set output layers
     * @param config model config parameters
     * @return true if success;false otherwise
    */
    bool LOWLIGHTSnpe::Initialize(const ObjectDetectionSnpeConfig& config)
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
        if (!m_snperuntime->Initialize(config.model_path, config.runtime)) 
        {
            LOG_ERROR("Failed to Initialize snpe instance.\n");
            return false;
        }

        m_isInit = true;
        return true;
    }


    /** @brief To deallocate buffers and reset
    */
    bool LOWLIGHTSnpe::DeInitialize()
    {
        if (m_isInit) 
        {
            m_snperuntime->Deinitialize();
            m_snperuntime.reset(nullptr);
        }

        m_isInit = false;
        return true;
    }

    bool LOWLIGHTSnpe::IsInitialized() const 
    {
        return m_isInit;
    }

    /** @brief To preprocess input image
     * @param input_image Input image for inference
     * @return true if succuess; false otherwise
    */
    bool LOWLIGHTSnpe::PreProcessInput(const cv::Mat& input_image,string model_name)
    {
        if (input_image.empty()) {
            LOG_ERROR("Invalid image!\n");
            return false;
        }

        auto inputShape = m_snperuntime->GetInputShape(m_inputLayers[0]);
        int model_h = inputShape[1];
        int model_w = inputShape[2];
        int channels = inputShape[3];

        cv::Mat image(model_h, model_w, CV_32FC3,cv::Scalar(0.0));
        cv::resize(input_image,image,cv::Size(model_h,model_w),cv::INTER_CUBIC);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        cv::Mat input(model_h, model_w, CV_32FC3, cv::Scalar(0.0));
        image.convertTo(input, CV_32FC3,1.0);

        vector<float> app_vect;
        
        if (input.isContinuous()) 
        {
            app_vect.assign((float*)input.data, (float*)input.data + input.total()*input.channels());
        } 
        else 
        {
            for (int i = 0; i < input.rows; ++i)
            {
                app_vect.insert(app_vect.end(), input.ptr<float>(i), input.ptr<float>(i)+input.cols*input.channels());
            }
        }

        float ***app = new float**[model_w];
        for (int i = 0; i < model_w; ++i) 
        {
            app[i] = new float*[model_h];
            for (int j = 0; j < model_h; ++j)
                app[i][j] = new float[channels];
        }

        for(int i = 0;i<model_w;i++)
        {
            for(int j=0;j<model_h;j++)
            {
                for(int k=0;k<channels;k++)
                {
                    app[i][j][k] = app_vect[ (i*model_h+j)*3  + k ];
                }
            }
        }

        float *input_tensor = NULL;
        input_tensor = m_snperuntime->GetInputTensor(m_inputLayers[0]);
        if (input_tensor == nullptr) {
            LOG_ERROR("Empty input tensor\n");
            return false;
        }

        float* pdata = (float*)(input.data);
        for(int i = 0;i<channels;i++)
        {
            for (int j = 0; j < model_w; j++)
            {
                for (int k = 0; k < model_h; k++)
                {
                    float x = app[j][k][i] / 255.0;
                    *pdata = x;
                    *input_tensor = x;
                    pdata += 1;
                    input_tensor += 1;
                }
            }
        }
        for (int i = 0; i < model_w; ++i) 
        {
            for (int j = 0; j < model_h; ++j) 
            {
                delete [] app[i][j];
            }
            delete [] app[i];
        }
        delete [] app;
        app = NULL;
        return true;
    }

    /** @brief To preprocess,execute and postprocess
     * @param input_image Input image for inference
     * @param output_image Inference output image
     * @param model_name To identify model for specific post-processing
     * @return true if success; false otherwise
    */
    bool LOWLIGHTSnpe::Detect(cv::Mat image,cv::Mat& output_image,string model_name)
    {
        /**
         * Preprocessing image
        */
        if(PreProcessInput(image, model_name) != true)
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
        if(PostProcess(output_image,model_name) != true)
        {
            LOG_ERROR("PostProcess failed\n");
            return false;
        }
        return true;
    }

    /** @brief Superres postprocess 
     * @param output_image Enhanced image
     * @param model_name To identify model for specific post-processing
    */
    bool LOWLIGHTSnpe::PostProcess(cv::Mat& output_image,string model_name)
    {
        auto outputShape = m_snperuntime->GetOutputShape(m_outputTensors[0]);
        float *predOutput =  m_snperuntime->GetOutputTensor(m_outputTensors[0]);

        if(predOutput == nullptr)
        {
            return false;
        }
        int height = outputShape[1];
        int width = outputShape[2];
        int channels = outputShape[3];
        
        cv::Mat temp0(cv::Size(width,height), CV_32FC3, predOutput);
        cv::cvtColor(temp0, temp0, cv::COLOR_RGB2BGR);

        vector<float> app_vect;
        
        if (temp0.isContinuous()) 
        {
            app_vect.assign((float*)temp0.data, (float*)temp0.data + temp0.total()*temp0.channels());
        } 
        else 
        {
            for (int i = 0; i < temp0.rows; ++i) 
            {
                app_vect.insert(app_vect.end(), temp0.ptr<float>(i), temp0.ptr<float>(i)+temp0.cols*temp0.channels());
            }
        }
        
        float ***app = new float**[channels];
        for (int i = 0; i < channels; ++i) 
        {
            app[i] = new float*[width];
            for (int j = 0; j < width; ++j)
                app[i][j] = new float[height];
        }
        
        for(int i = 0;i<channels;i++)
        {
            for(int j=0;j<width;j++)
            {
                for(int k=0;k<height;k++)
                {
                    app[i][j][k] = app_vect[i*width*height + j*height + k];
                }
            }
        }
        
        vector<float> app_t_vec;
        
        for(int i = 0;i<width;i++)
        {
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < channels; k++)
                {
                    float x;
                    if(model_name.compare("enhancementgan") == 0)
                    {
                        x = ((app[k][i][j] + 1)/2) * 255.0;
                    }
                    else
                    {
                        x = app[k][i][j] * 255.0;
                    }
                    
                    if(x>255.0)
                        x = 255.0;
                    app_t_vec.push_back(x);
                }
            }
        }

        output_image = cv::Mat(width, height, CV_32FC3,cv::Scalar(0.0));
        float* pdata = (float*)(output_image.data);
        for (int i = 0; i < channels*width*height; i++)
        {
            float x = app_t_vec[i];
            *pdata = x;
            pdata += 1;        
        }
        output_image.convertTo(output_image,CV_8UC3);
        
        for (int i = 0; i < channels; ++i) 
        {
            for (int j = 0; j < width; ++j) 
            {
                delete [] app[i][j];
            }
            delete [] app[i];
        }
        delete [] app;
        app = NULL;
        
        return true;
    }

} // namespace lowlightsnpe
