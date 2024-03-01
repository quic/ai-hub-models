// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "Configuration.h"
#include "Utils.h"
#include <map>

/** @brief To convert runtime from string to int 
 * @param device which contains runtime as a string
 * @return int value corresponding to runtime
*/

static runtime_t device2runtime(std::string&& device)
{
    /**
     * To convert all characters to lower case
    */
    std::transform(device.begin(), device.end(), device.begin(),
        [](unsigned char ch){ return tolower(ch); });

    if (0 == device.compare("dsp")) 
    {
        return DSP;
    }
    else 
    { 
        return CPU;
    }
}

/** @brief To parse Input config from config file
 * @param input contains input config array
*/
int Configuration::LoadInputConfig(Json::Value& input) 
{
     if (input.isArray()) 
     {
        int size = input.size();
            for (int i = 0; i < size; ++i) 
            {
                std::shared_ptr<InputConfiguration> inputconfig = std::shared_ptr<InputConfiguration>(new InputConfiguration());
                inputconfig->ConfigName = input[i][pipeline_input_config].asString();
                inputconfig->StreamType = input[i][stream_type].asString();
                inputconfig->Url = input[i][camera_url].asString();
                inputconfig->SkipFrame = input[i][skipframe].asInt();
                inputconfigs[inputconfig->ConfigName] = inputconfig;
            }
     }
     LOG_INFO("Input streams size=%u \n", input.size());
     return 0;
}

/** @brief To parse model config
 * @param models contains model config array
 */

int Configuration::LoadModelsConfig(Json::Value& models) 
{
    std::string line;
    if (models.isArray()) 
    {
        int size = models.size();
            for (int i = 0; i < size; ++i) 
            {
                std::shared_ptr<ObjectDetectionSnpeConfig> modelconfig = 
                std::shared_ptr<ObjectDetectionSnpeConfig>(new ObjectDetectionSnpeConfig());
                modelconfig->model_name = models[i][model_config_name].asString();
                modelconfig->model_type = models[i][model_type].asString();
                modelconfig->model_path = models[i][model_path].asString();
                modelconfig->runtime = device2runtime(models[i][runtime].asString());
                modelconfig->nmsThresh = models[i][nms_threshold].asFloat();
                modelconfig->confThresh = models[i][conf_threshold].asFloat();

                /**
                 * To access input layer names from config 
                */
                if (models[i]["input-layers"].isArray()) {
                    int num = models[i]["input-layers"].size();
                    for (int  j= 0; j < num; j++) {
                        modelconfig->inputLayers.push_back(models[i]["input-layers"][j].asString());
                    }
                }
                /**
                 * To access output layer names from config 
                */
                if (models[i][output_layers].isArray()) {
                    int num = models[i]["output-layers"].size();
                    for (int j = 0; j < num; j++) {
                        modelconfig->outputLayers.push_back(models[i]["output-layers"][j].asString());
                    }
                }
                /**
                 * To access output tensor names from config 
                */
                if (models[i][output_tensors].isArray()) {
                    int num = models[i]["output-tensors"].size();
                    for (int j = 0; j < num; j++) {
                        modelconfig->outputTensors.push_back(models[i]["output-tensors"][j].asString());
                    }
                }
                
                modelsconfig[modelconfig->model_name] = modelconfig;
            }
        }
        
        LOG_INFO("modelsconfig size = %lu \n", modelsconfig.size());
        return 0;
}

/** @brief To parse solution config
 * @param solutions contains solution array
 * 
*/

int Configuration::LoadSolutionsConfig(Json::Value& solutions) {
    if (solutions.isArray()) {
        int size = solutions.size();
            for (int i = 0; i < size; ++i) {
                std::shared_ptr<SolutionConfiguration> solutionconfig = std::shared_ptr<SolutionConfiguration>(new SolutionConfiguration());
                solutionconfig->solution_name = solutions[i][solution_name].asString();
                solutionconfig->model_name = solutions[i][model_name].asString();
                solutionconfig->Enable = solutions[i][Enable].asBool();
                solutionconfig->input_config_name = solutions[i][solution_input_config].asString();
                solutionconfig->output_type = solutions[i][output_type].asString();
                solutionsconfig[i] = solutionconfig;
            }
     }
     LOG_DEBUG("Solutions size %lu", solutionsconfig.size() );
     return 0;
}


/** @brief To parse config file
 * @param configFilePath contains json file passed as an argument 
*/
void Configuration::LoadConfiguration(string configFilePath)
{
    Json::Reader reader;
    Json::Value root;
    std::ifstream in(configFilePath, std::ios::binary);
    reader.parse(in, root);

    LoadInputConfig(root[input_configs]);
    LoadModelsConfig(root[model_configs]);
    LoadSolutionsConfig(root[solution_configs]);
}