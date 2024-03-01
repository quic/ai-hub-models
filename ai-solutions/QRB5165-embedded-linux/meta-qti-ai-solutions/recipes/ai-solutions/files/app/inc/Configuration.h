// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "Utils.h"

using namespace cv;
using namespace std;

const string input_configs = "input-configs";
const string model_configs = "model-configs";
const string solution_configs = "solution-configs";

// Input Configs;
const string pipeline_input_config = "input-config-name";
const string stream_type = "stream-type";
const string camera_url = "camera-url";
const string skipframe = "SkipFrame";

// Model Configs
const string model_config_name = "model-name";
const string model_type = "model-type";
const string model_path = "model-path";
const string runtime = "runtime";
const string nms_threshold = "nms-threshold";
const string conf_threshold = "conf-threshold";
const string input_layers = "input-layers";
const string output_layers = "output-layers";
const string output_tensors = "output-tensors";

// Solution Configs
const string solution_name = "solution-name";
const string model_name = "model-name";
const string Enable = "Enable";
const string solution_input_config = "input-config-name";
const string output_type = "output-type";

class ObjectDetectionSnpeConfig {
    public:
    string model_name;
    string model_type;
    std::string model_path;
    runtime_t runtime;
    float nmsThresh;
    float confThresh;
    std::vector<std::string> labels;
    std::vector<std::string> inputLayers;
    std::vector<std::string> outputLayers;
    std::vector<std::string> outputTensors;
};

class InputConfiguration{
    public:
    int SkipFrame;
    int StreamNumber=0;
    string StreamType;
    string Url;
    string ConfigName;
};

class SolutionConfiguration {
    public:
        string solution_name;
        string model_name;
        string input_config_name;
        bool Enable;
        string output_type;
        std::shared_ptr<InputConfiguration> input_config;
        std::shared_ptr<ObjectDetectionSnpeConfig> model_config;
};

class DebugConfiguration
{
    public:
    bool DumpData=false;
    string Directory;
};

class Configuration
{
public:
    static Configuration &getInstance()
    {
        static Configuration instance;
        return instance;
    }

private:
    Configuration() {}
public:
    Configuration(Configuration const &) = delete;
    void operator=(Configuration const &) = delete;

    DebugConfiguration Debug;
    ObjectDetectionSnpeConfig Config;
    SolutionConfiguration Sol_Config;
    std::unordered_map<std::string, std::shared_ptr<InputConfiguration>> inputconfigs;
    std::unordered_map<std::string, std::shared_ptr<ObjectDetectionSnpeConfig>> modelsconfig;
    std::unordered_map<int, std::shared_ptr<SolutionConfiguration>> solutionsconfig;

    void LoadConfiguration(string file);
    int LoadInputConfig(Json::Value& input);
    int LoadModelsConfig(Json::Value& models);
    int LoadSolutionsConfig(Json::Value& solutions);
};

#endif
