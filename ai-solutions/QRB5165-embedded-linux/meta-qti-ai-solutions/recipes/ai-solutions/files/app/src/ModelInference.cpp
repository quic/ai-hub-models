// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "ModelInference.h"
#include "Configuration.h"
#include "SuperresolutionSnpe.h"
#include "DetectionSnpe.h"
#include "LowlightSnpe.h"
#include "SegmentationSnpe.h"

using namespace std;
using namespace cv;
using namespace superressnpe;
using namespace detectionsnpe;
using namespace lowlightsnpe;
using namespace segmentationsnpe;

/** @brief contructor
*/
ModelInference::ModelInference()
{
        Impl = new SUPERRESSnpe();
}

/** @brief Parameter constructor
 * @param model_type To check model type from config file
*/
ModelInference::ModelInference(const string model_type)
{
    if (model_type.compare("superresolution") == 0) {            
        Impl = new SUPERRESSnpe();
        Model = SUPERRESOLUTION;
    }
    else if(model_type.compare("detection") == 0)
    {
        Impl = new DETECTIONSnpe();
        Model = DETECTION;
    }
    else if(model_type.compare("lowlight") == 0)
    {
        Impl = new LOWLIGHTSnpe();
        Model = LOWLIGHT;
    }
    else if(model_type.compare("segmentation") == 0)
    {
        Impl = new SEGMENTATIONSnpe();
        Model = SEGMENTATION;
    }
    else
        LOG_ERROR("Model implementation not found\n");  

    LOG_INFO("Initialized model = %s \n", model_type.c_str());

}

/** @brief destructor
*/
ModelInference::~ModelInference()
{
    if (nullptr != Impl) 
    {
        if (Model == SUPERRESOLUTION) 
        {    
            delete static_cast<SUPERRESSnpe*>(Impl);
        }
        else if(Model == DETECTION)
        {
            delete static_cast<DETECTIONSnpe*>(Impl);
        }
        else if(Model == LOWLIGHT)
        {
            delete static_cast<LOWLIGHTSnpe*>(Impl);
        }        
        else if(Model == SEGMENTATION)
        {
            delete static_cast<SEGMENTATIONSnpe*>(Impl);   
        }
        Impl = nullptr;
    }
}

/** @brief For model inference 
 * @param item contains image buffer and results object to store results
 * @return true if success
*/
int ModelInference::Inference(cv::Mat input,cv::Mat& output_image,string model_name)
{
    int ret=0;
    if (nullptr != Impl && IsInitialized()) 
    {
        if (Model == SUPERRESOLUTION) 
        {    
            ret = static_cast<SUPERRESSnpe*>(Impl)->Detect(input,output_image, model_name);
        }
        else if(Model == DETECTION)
        {
            ret = static_cast<DETECTIONSnpe*>(Impl)->Detect(input, output_image,model_name);
        }
        else if(Model == LOWLIGHT)
        {
            ret = static_cast<LOWLIGHTSnpe*>(Impl)->Detect(input, output_image,model_name);
        }        
        else if(Model == SEGMENTATION)
        {
            ret = static_cast<SEGMENTATIONSnpe*>(Impl)->Detect(input,output_image, model_name);
        }
    } 
    return ret;
}

/** @brief To intialize SNPE
 * @param contains SNPE configuration
 * @return true if success 
*/
int ModelInference::Initialization(const ObjectDetectionSnpeConfig& config)
{
    int ret=0;
    if (IsInitialized()) {
        if (Model == SUPERRESOLUTION) 
        {    
            ret = static_cast<SUPERRESSnpe*>(Impl)->DeInitialize() && static_cast<SUPERRESSnpe*>(Impl)->Initialize(config);
        }
        else if(Model == DETECTION)
        {
            ret = static_cast<DETECTIONSnpe*>(Impl)->DeInitialize() && static_cast<DETECTIONSnpe*>(Impl)->Initialize(config);
        }
        else if(Model == LOWLIGHT)
        {
            ret = static_cast<LOWLIGHTSnpe*>(Impl)->DeInitialize() && static_cast<LOWLIGHTSnpe*>(Impl)->Initialize(config);
        }        
        else if(Model == SEGMENTATION)
        {
            ret = static_cast<SEGMENTATIONSnpe*>(Impl)->DeInitialize() && static_cast<SEGMENTATIONSnpe*>(Impl)->Initialize(config);
        }  
    } 
    else 
    {
        if (Model == SUPERRESOLUTION) 
        {    
            ret = static_cast<SUPERRESSnpe*>(Impl)->Initialize(config);
        }
        else if(Model == DETECTION)
        {
            ret = static_cast<DETECTIONSnpe*>(Impl)->Initialize(config);
        }
        else if(Model == LOWLIGHT)
        {
            ret = static_cast<LOWLIGHTSnpe*>(Impl)->Initialize(config);
        }        
        else if(Model == SEGMENTATION)
        {
            ret = static_cast<SEGMENTATIONSnpe*>(Impl)->Initialize(config);
        } 
    }
    return ret;
}

/** @brief To uninitialize SNPE
 * @return true if success
*/
bool ModelInference::UnInitialization()
{
    bool ret=false;
    if (nullptr != Impl && IsInitialized()) 
    {
        if (Model == SUPERRESOLUTION) 
        {    
            ret = static_cast<SUPERRESSnpe*>(Impl)->DeInitialize();
        }
        else if(Model == DETECTION)
        {
            ret = static_cast<DETECTIONSnpe*>(Impl)->DeInitialize();
        }
        else if(Model == LOWLIGHT)
        {
            ret = static_cast<LOWLIGHTSnpe*>(Impl)->DeInitialize();
        }        
        else if(Model == SEGMENTATION)
        {
            ret = static_cast<SEGMENTATIONSnpe*>(Impl)->DeInitialize();
        } 
    } 
    else 
    {
        LOG_ERROR("ObjectDetection: deinit failed!\n");
        ret = false;
    }
    return ret;
}

/** @brief To check if SNPE is initialized
 * @return true if already inititalized
*/
bool ModelInference::IsInitialized()
{
    bool ret=false;
    if (Model == SUPERRESOLUTION) 
    {    
        ret = static_cast<SUPERRESSnpe*>(Impl)->IsInitialized();
    }
    else if(Model == DETECTION)
    {
        ret = static_cast<DETECTIONSnpe*>(Impl)->IsInitialized();
    }
    else if(Model == LOWLIGHT)
    {
        ret = static_cast<LOWLIGHTSnpe*>(Impl)->IsInitialized();
    }        
    else if(Model == SEGMENTATION)
    {
        ret = static_cast<SEGMENTATIONSnpe*>(Impl)->IsInitialized();
    } 
    return ret;
}

