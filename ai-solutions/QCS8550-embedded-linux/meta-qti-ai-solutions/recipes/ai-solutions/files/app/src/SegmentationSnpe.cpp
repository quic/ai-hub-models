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
#include "SegmentationSnpe.h"

namespace segmentationsnpe 
{

    /** @brief Constructor
    */
    SEGMENTATIONSnpe::SEGMENTATIONSnpe() :  m_isInit(false),m_snperuntime(nullptr) 
    {

    }

    /** @brief Destructor
    */
    SEGMENTATIONSnpe::~SEGMENTATIONSnpe() 
    {
        DeInitialize();
    }

    /** @brief To read model config and set output layers
     * @param config model config parameters
     * @return true if success;false otherwise
    */
    bool SEGMENTATIONSnpe::Initialize(const ObjectDetectionSnpeConfig& config)
    {
        m_snperuntime = std::move(std::unique_ptr<snperuntime::SNPERuntime>(new snperuntime::SNPERuntime()));

        m_inputLayers = config.inputLayers;
        m_outputLayers = config.outputLayers;
        m_outputTensors = config.outputTensors;
        m_nmsThresh = config.nmsThresh;
        m_confThresh = config.confThresh;

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
    bool SEGMENTATIONSnpe::DeInitialize()
    {
        if (m_isInit) {
            m_snperuntime->Deinitialize();
            m_snperuntime.reset(nullptr);
        }

        m_isInit = false;
        return true;
    }

    bool SEGMENTATIONSnpe::SetScoreThresh(const float& conf_thresh, const float& nms_thresh = 0.5) 
    {
            this->m_nmsThresh  = nms_thresh;
            this->m_confThresh = conf_thresh;
            return true;
    }

    bool SEGMENTATIONSnpe::IsInitialized() const 
    {
        return m_isInit;
    }

    /** @brief To preprocess input image
     * @param input_image Input image for inference
     * @return true if succuess; false otherwise
    */
    bool SEGMENTATIONSnpe::PreProcessInput(const cv::Mat& input_image,string model_name)
    {
        if (input_image.empty()) {
            LOG_ERROR("Invalid image!\n");
            return false;
        }

        auto inputShape = m_snperuntime->GetInputShape(m_inputLayers[0]);
        int model_h = inputShape[1];
        int model_w = inputShape[2];

        if (m_snperuntime->GetInputTensor(m_inputLayers[0]) == nullptr) 
        {
            LOG_ERROR("Empty input tensor\n");
            return false;
        }

        cv::Mat image = cv::Mat(model_h,model_w, CV_32FC3, Scalar(0.));
        cv::resize(input_image,image,cv::Size(model_h,model_w));
        cv::Mat input(model_h, model_w, CV_32FC3,  m_snperuntime->GetInputTensor(m_inputLayers[0]));

        if(model_name.compare("DeepLabv3Plus-resnet++") == 0 || model_name.compare("DeepLabv3-resnet101") == 0 || model_name.compare("DeepLabv3-resnet50") == 0 || model_name.compare("FCN_resnet101") == 0 || model_name.compare("FCN_resnet50") == 0)
        {
            cv::resize(image,image,cv::Size(model_w,model_h));
            image.convertTo(input,CV_32FC3,1.0);
            const float mean_vals[3] = {0.485, 0.456, 0.406};
            const float norm_vals[3] = {0.229, 0.224, 0.225};
            for (int i = 0; i < input.rows; i++)
            {
                float* pdata = (float*)(input.data + i * input.step);
                for (int j = 0; j < input.cols; j++)
                { 
                    float x = pdata[2], y=pdata[1], z = pdata[0];
                    pdata[0] = (x / 255.0  - mean_vals[0]) / norm_vals[0];
                    pdata[1] = (y / 255.0 - mean_vals[1]) / norm_vals[1];
                    pdata[2] = (z / 255.0 - mean_vals[2]) / norm_vals[2];
                    pdata += 3;
                }
            }
        }
        return true;
    }

    /** @brief To preprocess,execute and postprocess
     * @param input_image Input image for inference
     * @param output_image Inference output image
     * @param model_name To identify model for specific post-processing
     * @return true if success; false otherwise
    */
    bool SEGMENTATIONSnpe::Detect(cv::Mat image,cv::Mat& output_image,string model_name)
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
        if(PostProcess(image,output_image,model_name) != true)
        {
            LOG_ERROR("PostProcess failed\n");
            return false;
        }
        return true;
    }

    /** @brief postprocess to overlay segmentation
     * @param output_image Overlayed image
     * @param model_name To identify model for specific post-processing
    */
    bool SEGMENTATIONSnpe::PostProcess( cv::Mat image,cv::Mat& output_image,string model_name)
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

        cv::Mat temp = cv::Mat(height,width, CV_8UC3);
        vector<float> app_vect;

        float ***app = new float**[height];
        for (int i = 0; i < height; ++i) 
        {
            app[i] = new float*[width];
            for (int j = 0; j < width; ++j)
                app[i][j] = new float[channels];
        }
        
        for(int i = 0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {
                for(int k=0;k<channels;k++)
                {
                    app[i][j][k] = predOutput[i*channels*height + j*channels + k];
                }
            }
        }   

        vector<float> app_t_vec;
        
        for(int i = 0;i < channels;i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < height; k++)
                {
                    float x = app[j][k][i];
                    app_t_vec.push_back(x);
                }
            }
        }

        float ***app_t=NULL;

        app_t = new float**[channels];
        for (int i = 0; i < channels; ++i) 
        {
            app_t[i] = new float*[width];
            for (int j = 0; j < width; ++j)
                app_t[i][j] = new float[height];
        }

        for(int i =0;i<channels; i++)
        {
            for(int j=0;j<width;j++)
            {
                for(int k=0;k<height;k++)
                {
                    app_t[i][j][k]=app_t_vec[i*height*width + j*height + k];
                }
            }
        }

        if(model_name.compare("DeepLabv3Plus-resnet++") == 0)
        {
            vector<vector<int>> colors_res = {
                                                {  0,   0,   0},{128,   0,   0},{  0, 128,   0},{128, 128,   0},{  0,   0, 128},
                                                {128,   0, 128},{  0, 128, 128},{128, 128, 128},{ 64,   0,   0},{192,   0,   0},
                                                { 64, 128,   0},{192, 128,   0},{ 64,   0, 128},{192,   0, 128},{ 64, 128, 128},
                                                {192, 128, 128},{  0,  64,   0},{128,  64,   0},{  0, 192,   0},{128, 192,   0},
                                                {  0,  64, 128},{128,  64, 128},{  0, 192, 128},{128, 192, 128},{ 64,  64,   0},
                                                {192,  64,   0},{ 64, 192,   0},{192, 192,   0},{ 64,  64, 128},{192,  64, 128},
                                                { 64, 192, 128},{192, 192, 128},{  0,   0,  64},{128,   0,  64},{  0, 128,  64},
                                                {128, 128,  64},{  0,   0, 192},{128,   0, 192},{  0, 128, 192},{128, 128, 192},
                                                { 64,   0,  64},{192,   0,  64},{ 64, 128,  64},{192, 128,  64},{ 64,   0, 192},
                                                {192,   0, 192},{ 64, 128, 192},{192, 128, 192},{  0,  64,  64},{128,  64,  64},
                                                {  0, 192,  64},{128, 192,  64},{  0,  64, 192},{128,  64, 192},{  0, 192, 192},
                                                {128, 192, 192},{ 64,  64,  64},{192,  64,  64},{ 64, 192,  64},{192, 192,  64},
                                                { 64,  64, 192},{192,  64, 192},{ 64, 192, 192},{192, 192, 192},{ 32,   0,   0},
                                                {160,   0,   0},{ 32, 128,   0},{160, 128,   0},{ 32,   0, 128},{160,   0, 128},
                                                { 32, 128, 128},{160, 128, 128},{ 96,   0,   0},{224,   0,   0},{ 96, 128,   0},
                                                {224, 128,   0},{ 96,   0, 128},{224,   0, 128},{ 96, 128, 128},{224, 128, 128},
                                                { 32,  64,   0},{160,  64,   0},{ 32, 192,   0},{160, 192,   0},{ 32,  64, 128},
                                                {160,  64, 128},{ 32, 192, 128},{160, 192, 128},{ 96,  64,   0},{224,  64,   0},
                                                { 96, 192,   0},{224, 192,   0},{ 96,  64, 128},{224,  64, 128},{ 96, 192, 128},
                                                {224, 192, 128},{ 32,   0,  64},{160,   0,  64},{ 32, 128,  64},{160, 128,  64},
                                                { 32,   0, 192},{160,   0, 192},{ 32, 128, 192},{160, 128, 192},{ 96,   0,  64},
                                                {224,   0,  64},{ 96, 128,  64},{224, 128,  64},{ 96,   0, 192},{224,   0, 192},
                                                { 96, 128, 192},{224, 128, 192},{ 32,  64,  64},{160,  64,  64},{ 32, 192,  64},
                                                {160, 192,  64},{ 32,  64, 192},{160,  64, 192},{ 32, 192, 192},{160, 192, 192},
                                                { 96,  64,  64},{224,  64,  64},{ 96, 192,  64},{224, 192,  64},{ 96,  64, 192},
                                                {224,  64, 192},{ 96, 192, 192},{224, 192, 192},{  0,  32,   0},{128,  32,   0},
                                                {  0, 160,   0},{128, 160,   0},{  0,  32, 128},{128,  32, 128},{  0, 160, 128},
                                                {128, 160, 128},{ 64,  32,   0},{192,  32,   0},{ 64, 160,   0},{192, 160,   0},
                                                { 64,  32, 128},{192,  32, 128},{ 64, 160, 128},{192, 160, 128},{  0,  96,   0},
                                                {128,  96,   0},{  0, 224,   0},{128, 224,   0},{  0,  96, 128},{128,  96, 128},
                                                {  0, 224, 128},{128, 224, 128},{ 64,  96,   0},{192,  96,   0},{ 64, 224,   0},
                                                {192, 224,   0},{ 64,  96, 128},{192,  96, 128},{ 64, 224, 128},{192, 224, 128},
                                                {  0,  32,  64},{128,  32,  64},{  0, 160,  64},{128, 160,  64},{  0,  32, 192},
                                                {128,  32, 192},{  0, 160, 192},{128, 160, 192},{ 64,  32,  64},{192,  32,  64},
                                                { 64, 160,  64},{192, 160,  64},{ 64,  32, 192},{192,  32, 192},{ 64, 160, 192},
                                                {192, 160, 192},{  0,  96,  64},{128,  96,  64},{  0, 224,  64},{128, 224,  64},
                                                {  0,  96, 192},{128,  96, 192},{  0, 224, 192},{128, 224, 192},{ 64,  96,  64},
                                                {192,  96,  64},{ 64, 224,  64},{192, 224,  64},{ 64,  96, 192},{192,  96, 192},
                                                { 64, 224, 192},{192, 224, 192},{ 32,  32,   0},{160,  32,   0},{ 32, 160,   0},
                                                {160, 160,   0},{ 32,  32, 128},{160,  32, 128},{ 32, 160, 128},{160, 160, 128},
                                                { 96,  32,   0},{224,  32,   0},{ 96, 160,   0},{224, 160,   0},{ 96,  32, 128},
                                                {224,  32, 128},{ 96, 160, 128},{224, 160, 128},{ 32,  96,   0},{160,  96,   0},
                                                { 32, 224,   0},{160, 224,   0},{ 32,  96, 128},{160,  96, 128},{ 32, 224, 128},
                                                {160, 224, 128},{ 96,  96,   0},{224,  96,   0},{ 96, 224,   0},{224, 224,   0},
                                                { 96,  96, 128},{224,  96, 128},{ 96, 224, 128},{224, 224, 128},{ 32,  32,  64},
                                                {160,  32,  64},{ 32, 160,  64},{160, 160,  64},{ 32,  32, 192},{160,  32, 192},
                                                { 32, 160, 192},{160, 160, 192},{ 96,  32,  64},{224,  32,  64},{ 96, 160,  64},
                                                {224, 160,  64},{ 96,  32, 192},{224,  32, 192},{ 96, 160, 192},{224, 160, 192},
                                                { 32,  96,  64},{160,  96,  64},{ 32, 224,  64},{160, 224,  64},{ 32,  96, 192},
                                                {160,  96, 192},{ 32, 224, 192},{160, 224, 192},{ 96,  96,  64},{224,  96,  64},
                                                { 96, 224,  64},{224, 224,  64},{ 96,  96, 192},{224,  96, 192},{ 96, 224, 192},
                                                {224, 224, 192}
                                            };

                int  **app_t_max=NULL;
                
                app_t_max = new int*[width];
                for (int j = 0; j < width; ++j)
                {
                    app_t_max[j] = new int[height];
                }
                
                vector<float> max_values;
                for(int i=0;i<height;i++)
                {
                    for(int j=0;j<width;j++)
                    {
                        float max = app_t[0][i][j];
                        app_t_max[i][j] = 0;
                        for(int k=1; k<channels;k++)
                        {
                            float temp = app_t[k][i][j];
                            if( temp > max)
                            {
                                max = temp;
                                app_t_max[i][j] = k;
                            }
                        }
                        max_values.push_back(max);    
                    }
                }

                vector<float> max_vec;

                for(int i = 0; i< height;i++)
                {
                    for(int j=0;j<width;j++)
                    {
                        max_vec.push_back(app_t_max[i][j]);
                    }
                }
                
                vector<vector<int>> color;
                color = colors_res;
                
                for (int i = 0; i < temp.rows; i++)
                {
                    char* pdata = (char*)(temp.data + i * temp.step);
                    for (int j = 0; j < temp.cols; j++)
                    { 
                        int id = app_t_max[i][j];
                        pdata[0] = color[id][2];
                        pdata[1] = color[id][1];
                        pdata[2] = color[id][0];
                        pdata += 3;
                    }
                }

                for (int j = 0; j < width; ++j) 
                {
                    delete [] app_t_max[j];
                }
                delete [] app_t_max;
                app_t_max = NULL;

        }
        else if(model_name.compare("DeepLabv3-resnet101") == 0 || model_name.compare("DeepLabv3-resnet50") == 0 || model_name.compare("FCN_resnet101") == 0 || model_name.compare("FCN_resnet50") == 0)
        {

                vector<vector<int>> label_map = {
                                                    {0, 0, 0},  // background
                                                    {128, 0, 0}, // aeroplane
                                                    {0, 128, 0}, // bicycle
                                                    {128, 128, 0}, // bird
                                                    {0, 0, 128}, // boat
                                                    {128, 0, 128}, // bottle
                                                    {0, 128, 128}, // bus
                                                    {128, 128, 128}, // car
                                                    {64, 0, 0}, // cat
                                                    {192, 0, 0}, // chair
                                                    {64, 128, 0}, // cow
                                                    {192, 128, 0}, // dining table
                                                    {64, 0, 128}, // dog
                                                    {192, 0, 128}, // horse
                                                    {64, 128, 128}, // motorbike
                                                    {192, 128, 128}, // person
                                                    {0, 64, 0}, // potted plant
                                                    {128, 64, 0}, // sheep
                                                    {0, 192, 0}, // sofa
                                                    {128, 192, 0}, // train
                                                    {0, 64, 128} // tv/monitor
                                                };

                int **app_t_max=NULL;
                
                app_t_max = new int*[width];
                for (int j = 0; j < width; j++)
                {
                    app_t_max[j] = new int[height];
                }
                
                vector<float> max_values;
                for(int i=0; i<height; i++)
                {
                    for(int j=0; j<width; j++)
                    {
                        float max = app_t[0][i][j];
                        app_t_max[i][j] = 0;
                        for(int k=1; k<channels;k++)
                        {
                            float temp = app_t[k][i][j];
                            if( temp > max)
                            {
                                max = temp;
                                app_t_max[i][j] = k;
                            }
                        }
                        max_values.push_back(max);
                        
                    }
                }

                vector<float> max_vec;

                for(int i = 0; i< height;i++)
                {
                    for(int j=0;j<width;j++)
                    {
                        max_vec.push_back(app_t_max[i][j]);   
                    }
                }
                
                for(size_t i=0; i< label_map.size();i++ )
                {
                    for (int i = 0; i < temp.rows; i++)
                    {
                        char* pdata = (char*)(temp.data + i * temp.step);
                        for (int j = 0; j < temp.cols; j++)
                        { 
                            int id = app_t_max[i][j];
                            pdata[0] = label_map[id][0];
                            pdata[1] = label_map[id][1];
                            pdata[2] = label_map[id][2];
                            pdata += 3;
                        }
                    }
                }
                
        }

        double alpha = 0.6;
        double beta = 1 - alpha ;
        double gamma = 0;

        cv::resize(image,image,cv::Size(height,width));
        temp.convertTo(temp, CV_8UC3, 1.0);
        cv::addWeighted(image, alpha, temp, beta, gamma, output_image);
        
        for (int i = 0; i < height; ++i) 
        {
            for (int j = 0; j < width; ++j) 
            {
                delete [] app[i][j];
            }
            delete [] app[i];
        }
        delete [] app;
        app = NULL;


        for (int i = 0; i < channels; ++i) 
        {
            for (int j = 0; j < width; ++j) 
            {
                delete [] app_t[i][j];
            }
            delete [] app_t[i];
        }
        delete [] app_t;
        app_t = NULL;
        
        return true;

    }

} // namespace segmentationsnpe