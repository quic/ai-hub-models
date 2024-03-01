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
#include "DetectionSnpe.h"

namespace detectionsnpe 
{

    /** @brief Constructor
    */
    DETECTIONSnpe::DETECTIONSnpe() : m_isInit(false),m_snperuntime(nullptr)
    {

    }

    /** @brief Destructor
    */
    DETECTIONSnpe::~DETECTIONSnpe() 
    {
        DeInitialize();
    }

    /** @brief To read model config and set output layers
     * @param config model config parameters
     * @return true if success;false otherwise
    */
    bool DETECTIONSnpe::Initialize(const ObjectDetectionSnpeConfig& config)
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
    bool DETECTIONSnpe::DeInitialize()
    {
        if (m_isInit) 
        {
            m_snperuntime->Deinitialize();
            m_snperuntime.reset(nullptr);
        }
        m_isInit = false;
        return true;
    }

    bool DETECTIONSnpe::SetScoreThresh(const float& conf_thresh, const float& nms_thresh = 0.5) 
    {
            this->m_nmsThresh  = nms_thresh;
            this->m_confThresh = conf_thresh;
            return true;
    }

    bool DETECTIONSnpe::IsInitialized() const 
    {
        return m_isInit;
    }

    /** @brief To preprocess input image
     * @param input_image Input image for inference
     * @return true if succuess; false otherwise
    */
    bool DETECTIONSnpe::PreProcessInput(const cv::Mat& input_image,string model_name)
    {
        if (input_image.empty()) {
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
        
        cv::Mat image = cv::Mat(model_h,model_w, CV_32FC3, Scalar(0.));
        cv::resize(input_image,image,cv::Size(model_h,model_w));
        cv::Mat input(model_h, model_w, CV_32FC3,  m_snperuntime->GetInputTensor(m_inputLayers[0]));

        if(model_name.compare("ssd-mobilenet-v2") == 0 )
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            image.convertTo(image, CV_32S);
            subtract(image,Scalar(123.0, 117.0, 104.0),image);
            image.convertTo(input, CV_32FC3, 1.0);
        }
        else if(model_name.compare("yolo-nas") == 0)
        {
            image.convertTo(input, CV_32FC3, 1/255.0);
        }
        else if(model_name.compare("yolo-x") == 0)
        {
            image.convertTo(input, CV_32FC3, 1.0);
        }

        return true;
    }

    /** @brief To preprocess,execute and postprocess
     * @param input_image Input image for inference
     * @param output_image Inference output image
     * @param model_name To identify model for specific post-processing
     * @return true if success; false otherwise
    */
    bool DETECTIONSnpe::Detect(cv::Mat image,cv::Mat& output_image,string model_name)
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
         * Postprocessing to extract bounding boxes
        */
        if(PostProcess(image,output_image,model_name) != true)
        {
            LOG_ERROR("PostProcess failed\n");
            return false;
        }
        return true;
    }
    
    float DETECTIONSnpe::computeIoU(const cv::Rect& a, const cv::Rect& b) 
    {
        float xOverlap = std::max(
            0.,
            std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x) + 1.);
        float yOverlap = std::max(
            0.,
            std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y) + 1.);
        float intersection = xOverlap * yOverlap;
        float unio =
            (a.width + 1.) * (a.height + 1.) +
            (b.width + 1.) * (b.height + 1.) - intersection;
        return intersection / unio;
    }

    std::vector<ObjectData> DETECTIONSnpe::doNMS(std::vector<ObjectData> winList, const float& nms_thresh) 
    {
            if (winList.empty()) {
                return winList;
            }

            std::sort(winList.begin(), winList.end(), [] (const ObjectData& left, const ObjectData& right) {
                if (left.confidence > right.confidence) {
                    return true;
                } else {
                    return false;
                }
            });

            std::vector<bool> flag(winList.size(), false);
            for (unsigned int i = 0; i < winList.size(); i++) {
                if (flag[i]) {
                    continue;
                }

                for (unsigned int j = i + 1; j < winList.size(); j++) {
                    if (computeIoU(winList[i].bbox, winList[j].bbox) > nms_thresh) {
                        flag[j] = true;
                    }
                }
            }

            std::vector<ObjectData> ret;
            for (unsigned int i = 0; i < winList.size(); i++) {
                if (!flag[i])
                    ret.push_back(winList[i]);
            }
            return ret;
        }

    /** @brief Object Detection postprocess 
     * @param output_image Image with bounding boxes
     * @param model_name To identify model for specific post-processing
    */
    bool DETECTIONSnpe::PostProcess( cv::Mat image,cv::Mat& output_image,string model_name)
    {
        int width = image.cols, height = image.rows;
        cv::resize(image,output_image,cv::Size(width,height));

        if(model_name.compare("ssd-mobilenet-v2") == 0)
        {
            vector<string>classes = {
                                        "background","aeroplane","bicycle","bird","boat",
                                        "bottle","bus","car","cat","chair","cow",
                                        "diningtable","dog","horse","motorbike","person",
                                        "pottedplant","sheep","sofa","train","tvmonitor",
                                    };

            auto outputShape_score = m_snperuntime->GetOutputShape(m_outputTensors[0]);
            int elements_score = outputShape_score[1];
            int channels_score = outputShape_score[2]; 

            auto outputShape_box = m_snperuntime->GetOutputShape(m_outputTensors[1]);
            float *score_confidence =  m_snperuntime->GetOutputTensor(m_outputTensors[0]);
            float *box_coordinates =  m_snperuntime->GetOutputTensor(m_outputTensors[1]);

            if( (score_confidence == nullptr) || (box_coordinates == nullptr) )
            {
                return false;
            }
            for(size_t class_index = 1; class_index<classes.size();class_index++)
            {
                std::vector<ObjectData> winList;
                for(int row=0; row<elements_score; row++)
                {
                    for(int col=0;col<channels_score;col++)
                    {
                        int element = channels_score*row + col;
                        float value = score_confidence[element];
                        long unsigned int class_pred = element%channels_score;
                        if(value > m_confThresh && (class_pred==class_index) )
                        {
                            ObjectData rect;
                            rect.bbox.x =  box_coordinates[row*4 ]  * width;
                            rect.bbox.y =  box_coordinates[row*4+ 1]  * height;
                            rect.bbox.width  =  box_coordinates[row*4 + 2]  * width;
                            rect.bbox.height =  box_coordinates[row*4 + 3]  * height;
                            rect.confidence = value;
                            rect.label = class_pred;
                            winList.push_back(rect);
                        }
                    }
                }
                winList = doNMS(winList, m_nmsThresh);
                for(size_t i =0;i<winList.size();i++)
                {
                    ObjectData result = winList[i];
                    cv::Point start = cv::Point(result.bbox.x, result.bbox.y);
                    cv::Point end = cv::Point(result.bbox.width, result.bbox.height);

                    cv::rectangle(output_image, start,end, cv::Scalar(255, 0, 0), 3);
                    cv::Point position = cv::Point(result.bbox.x, result.bbox.y - 5);
                    cv::putText(output_image, classes[result.label], position, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(255, 0, 0), 2, 0.3);
                }
            }
        }
        else if(model_name.compare("yolo-nas") == 0)
        {
            vector<string> classes = {
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic", "fire", "stop", "parking",
                                        "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                                        "sports", "kite", "baseball", "baseball", "skateboard", "surfboard",
                                        "tennis", "bottle", "wine", "cup", "fork", "knife","spoon",
                                        "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                        "carrot", "hot", "pizza", "donut", "cake", "chair", "couch",
                                        "potted", "bed", "dining", "toilet", "tv", "laptop", "mouse",
                                        "remote", "keyboard", "cell", "microwave", "oven", "toaster",
                                        "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy", "hair", "toothbrush"
                                    };

            float *class_scores =  m_snperuntime->GetOutputTensor(m_outputTensors[0]);
            auto outputShape_scores = m_snperuntime->GetOutputShape(m_outputTensors[0]);
            float *bboxes =  m_snperuntime->GetOutputTensor(m_outputTensors[1]);
            auto outputShape_bboxes = m_snperuntime->GetOutputShape(m_outputTensors[1]);

            if( (class_scores == nullptr) || (bboxes == nullptr) )
            {
                return false;
            }
            float ratio1 = width/320.0;
            float ratio2 = height/320.0;

            int out_coordinates = outputShape_scores[1];
            int out_scores = outputShape_scores[2];

            std::vector<ObjectData> winList;
            for(int i =0;i<out_coordinates;i++)
            {
                for(int j=0;j<out_scores;j++)
                {
                    if(class_scores[out_scores*i + j] >= m_confThresh)
                    {
                        float x1 = bboxes[i*4 ]*ratio1;
                        float y1 = bboxes[i*4 + 1]*ratio2;
                        float x2 = bboxes[i*4 + 2]*ratio1;
                        float y2 = bboxes[i*4 + 3]*ratio2;
                        ObjectData rect;
                        rect.bbox.x = x1 ;
                        rect.bbox.y = y1 ;
                        rect.bbox.width = x2 - x1;
                        rect.bbox.height = y2 - y1;
                        rect.confidence = class_scores[out_scores*i + j];
                        rect.label = j;
                        winList.push_back(rect);
                    }
                }
            }
            winList = doNMS(winList,m_nmsThresh);
            for(size_t i =0;i<winList.size();i++)
            {
                ObjectData result = winList[i];
                cv::rectangle(output_image, cv::Rect(result.bbox.x,result.bbox.y, result.bbox.width , result.bbox.height), cv::Scalar(255, 0, 0), 3);
                cv::Point position = cv::Point(result.bbox.x, result.bbox.y - 10);
                cv::putText(output_image, classes[result.label], position, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(255, 0, 0), 2, 0.3);
            }
        }
        else if(model_name.compare("yolo-x") == 0)
        {
            vector<string> classes = {
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic", "fire", "stop", "parking",
                                        "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                                        "sports", "kite", "baseball", "baseball", "skateboard", "surfboard",
                                        "tennis", "bottle", "wine", "cup", "fork", "knife","spoon",
                                        "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                        "carrot", "hot", "pizza", "donut", "cake", "chair", "couch",
                                        "potted", "bed", "dining", "toilet", "tv", "laptop", "mouse",
                                        "remote", "keyboard", "cell", "microwave", "oven", "toaster",
                                        "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy", "hair", "toothbrush"
                                    };
                
            float *scores =  m_snperuntime->GetOutputTensor(m_outputTensors[0]);
            auto outputShape = m_snperuntime->GetOutputShape(m_outputTensors[0]);

            if(scores == nullptr)
            {
                return false;
            }
            int model_h = outputShape[1];
            int model_w = outputShape[2];
            float output[model_h][model_w];

            for(int i=0;i<model_h;i++)
            {
                for(int j=0;j<model_w;j++)
                {
                    output[i][j] = scores[model_w*i+j];
                }
            }
            
            static bool flag=false;
            static vector<int> grid;
            static vector<int> expanded_stride;
            static int sum=0;
            if(flag == false)
            {
                const int strides[3] = {8, 16, 32};
                int hsizes[3] = {80, 40, 20};
                int wsizes[3] = {80, 40, 20};

                vector<vector<int>> grids, expanded_strides;
                
                for(int i=0;i<3;i++)
                {   
                    vector<int> grid;
                    vector<int> expanded_stride;
                    for(int j=0; j<hsizes[i];j++)
                    {   
                        for(int k=0;k<wsizes[i];k++)
                        {
                            grid.push_back(k);
                            grid.push_back(j);
                        }
                    }
                    for(int m=0;m<(hsizes[i]*hsizes[i]);m++)
                    {
                        expanded_stride.push_back(strides[i]);
                    }
                    grids.push_back(grid);
                    expanded_strides.push_back(expanded_stride);
                }

                int count = hsizes[0] * wsizes[0] * 2;
                for(int j=0;j<count;j++)
                {
                    grid.push_back(grids[0][j]);
                }
                count = hsizes[1] * wsizes[1] * 2;
                for(int j=0;j<count;j++)
                {
                    grid.push_back(grids[1][j]);
                }
                count = hsizes[2] * wsizes[2] * 2;
                for(int j=0;j<count;j++)
                {
                    grid.push_back(grids[2][j]);
                }
                
                count = hsizes[0] * wsizes[0];
                for(int j=0;j<count;j++)
                {
                    expanded_stride.push_back(expanded_strides[0][j]);
                }
                count = hsizes[1] * wsizes[1];
                for(int j=0;j<count;j++)
                {
                    expanded_stride.push_back(expanded_strides[1][j]);
                }
                count = hsizes[2] * wsizes[2];
                for(int j=0;j<count;j++)
                {
                    expanded_stride.push_back(expanded_strides[2][j]);
                }
                
                for(int i=0;i<3;i++)
                {
                    sum += hsizes[i] * wsizes[i];
                }
                flag = true;
            }
            
            for(int i=0; i<sum;i++)
            {
                for(int j=0;j<2;j++)
                {
                    output[i][j] = (output[i][j] + grid[2*i + j]) * expanded_stride[i];
                }
            }
            for(int i=0; i<sum;i++)
            {
                for(int j=2;j<4;j++)
                {
                    output[i][j] = exp(output[i][j]) * expanded_stride[i];
                }
            }

            vector<vector<int>>  boxes;
            vector<vector<float>> scores_vec;
            for(int i=0;i<sum;i++)
            {
                vector<int> box;
                for(int j=0;j<4;j++)
                {
                    box.push_back(output[i][j]);
                }
                boxes.push_back(box);
            }

            for(int i=0;i<sum;i++)
            {
                vector<float> score;
                float val = output[i][4];
                for(int j=5;j<85;j++)
                {
                    score.push_back(output[i][j] * val);
                }
                scores_vec.push_back(score);
            }
            
            std::vector<ObjectData> winList;
            for(int i=0;i<sum;i++)
            {
                cv::Mat classes_img(classes.size(),1,CV_32FC1);
                memcpy(classes_img.data, scores_vec[i].data(),classes.size()*sizeof(float));
                double minScore;
                double maxScore;
                Point minClassLoc;
                Point maxClassIndex;
                cv::minMaxLoc(classes_img, &minScore, &maxScore, &minClassLoc, &maxClassIndex);
                if(maxScore>=m_confThresh)
                {
                    for(int j=0;j<4;j++)
                    {
                        int x1 = boxes[i][0];
                        int y1 = boxes[i][1];
                        int x2 = boxes[i][2];
                        int y2 = boxes[i][3];

                        int x = (int)(x1 - x2/2);
                        int y = (int)(y1 - y2/2);
                        int w = (int)(x1 + x2/2);
                        int h = (int)(y1 + y2/2);

                        ObjectData rect;
                        float ratio1 = width/640.0;
                        float ratio2 = height/640.0;
                        rect.bbox.x = x * ratio1;
                        rect.bbox.y = y * ratio2;
                        rect.bbox.width = w *ratio1;
                        rect.bbox.height = h *ratio2;
                        rect.confidence = maxScore;
                        rect.label = maxClassIndex.y;

                        winList.push_back(rect);
                    }
                }
            }

            winList = doNMS(winList, m_nmsThresh);
            for(size_t i =0;i<winList.size();i++)
            {
                ObjectData result = winList[i];
                cv::rectangle(output_image, cv::Rect(result.bbox.x,result.bbox.y, result.bbox.width - result.bbox.x, result.bbox.height - result.bbox.y), cv::Scalar(255, 0, 0), 3);
                cv::Point position = cv::Point(result.bbox.x, result.bbox.y - 10);
                cv::putText(output_image, classes[result.label], position, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(255, 0, 0), 2, 0.3);
            }
        }
        return true;
    }

} // namespace detectionsnpe
