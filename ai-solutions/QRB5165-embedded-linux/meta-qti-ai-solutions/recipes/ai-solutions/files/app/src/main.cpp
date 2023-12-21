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
#include "StreamDecode.h"
#include "StreamEncode.h"
#include "DecodeQueue.h"
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;


/**
 * To decode frames from gstreamer 
*/
shared_ptr<DecodeQueue> gDecodeQueue; 

/**
 * To check for gstreamer exit
*/
bool gExit = false; 

/**
 * To encode frames for preview/file
*/
shared_ptr<EncodeController> encoderCtrl; 

/**
 * To create object for frame capture
*/
shared_ptr<CaptureController> captureCtrl; 



/** @brief To intialize and configure the runtime based on the solution
 * @param sol_conf contains information about the solution 
*/
void Inference_Image(void *sol_conf, string inputimage, string outputimage)
{
    LOG_DEBUG("InferenceThread \n");

    SolutionConfiguration *solution_config = (SolutionConfiguration *)sol_conf;
    /**
     * TO initialize layers and buffers based on model type
    */
    shared_ptr<ModelInference> shInference;
    shInference = std::make_shared<ModelInference>(solution_config->model_config->model_type);
    
    shInference->Initialization(*solution_config->model_config.get());
    /**
     * Run the loop until stream ends or interrupt from user
    */
    shared_ptr<DetectionItem> item;
    
    
    /**
     * start inferencing on the image buffer 
    */
    auto start1 = chrono::steady_clock::now();
    cv::Mat input = cv::imread(inputimage, cv::IMREAD_COLOR);
	if(input.empty())
	{
		LOG_ERROR("Invalid image!\n");
		return;
	}
    
    LOG_ERROR("model name = %s\n",solution_config->model_name.c_str());
    cv::Mat output_image;
    if(shInference->Inference(input,output_image,solution_config->model_name) == true)
    {
        auto end1 = chrono::steady_clock::now();
        auto costTime1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();
        LOG_INFO("Elapsed inference time in milliseconds: %ld ms\n",costTime1);
        cv::imwrite(outputimage,output_image);
    }
    else
    {
        LOG_ERROR("Model Inference failed\n");
    }
    shInference->UnInitialization();
}

void Inference_Camera(void *sol_conf)
{
    LOG_DEBUG("InferenceThread \n");

    SolutionConfiguration *solution_config = (SolutionConfiguration *)sol_conf;
    /**
     * TO initialize layers and buffers based on model type
    */
    shared_ptr<ModelInference> shInference;
    shInference = std::make_shared<ModelInference>(solution_config->model_config->model_type);
    
    shInference->Initialization(*solution_config->model_config.get());

    int ret = 0;
    auto start = chrono::steady_clock::now();
    uint32_t frames = 0;
    /**
     * Run the loop until stream ends or interrupt from user
    */
    do
    {
        shared_ptr<DetectionItem> item;
        /**
         * To retrieve gstreamer buffer from queue
        */
        ret = gDecodeQueue->Dequeue(item, 300);
        /**
         * Check if Dequeue is successful
        */
        if (ret == 0)
        {
            frames += 1;
            auto start1 = chrono::steady_clock::now();
            /**
             * start inferencing on the image buffer 
            */
            cv::Mat image(cv::Size(item->Width, item->Height), CV_8UC3, item->ImageBuffer.get(), cv::Mat::AUTO_STEP);
			if(image.empty())
			{
				LOG_ERROR("Invalid image!\n");
				return;
			}
            cv::Mat output_image;
            shInference->Inference(image,output_image,solution_config->model_name);
            auto end1 = chrono::steady_clock::now();
            auto costTime1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();
            LOG_INFO("Elapsed inference time in milliseconds: %ld ms\n",costTime1);

            cv::resize(output_image,output_image,Size(1280,720));
            int size = output_image.total() * output_image.elemSize();
            /**
             * To display on monitor
            */
            encoderCtrl->EncodeFrame(item->StreamId, output_image.data, size);
        }
        /**
         * If there are no items in the queue
        */
        else
        {
            if (ret != 1)
            {
                LOG_ERROR("Error ret= %d\n", ret);
            }
            continue;
        }

    } while (!gExit);
    /**
     * To infer on the remaining pending items if exited before completion
    */
    auto remains = gDecodeQueue->GetRemainItems();
    LOG_INFO("Remain Items= %lu\n", remains.size());
    for (auto item : remains)
    {
        frames += 1;
        cv::Mat image(cv::Size(item->Width, item->Height), CV_8UC3, item->ImageBuffer.get(), cv::Mat::AUTO_STEP);
        cv::Mat output_image;
        shInference->Inference(image,output_image,solution_config->model_name);
    }
    /**
     * To deallocate the bufferes and runtime
    */
    shInference->UnInitialization();

    auto end = chrono::steady_clock::now();
    auto costTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    LOG_INFO("Elapsed time in milliseconds: %ld ms \t Received Frames: %d \t Through rate: %ld \n", 
    costTime, frames, (frames * 1000)/costTime);
}

/** @brief Execution starts from here  
 * @param argc for total argument count
 * @param argv arguments to be passed
*/

int main(int argc, char **argv)
{
    /**
     * To store config file name passed in argument
    */
    const char* inputFile=NULL;
    string inputimage,outputimage;
    int opt = 0;
    /**
     * Check if 'h' or 'c' passed in argument
    */
    while ((opt = getopt(argc, argv, ":hc:i:o:")) != EOF) 
    {
        switch (opt)
        {
            case 'h': std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to run the use case\n"
                        << "using the SNPE C++ API.\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -c  <FILE>   Path to the config json file.\n"
                        << "Example: ai-solutions -c data/config.json -i image_path -o Output_path\n";
                        break;
            case 'c':
                    inputFile = optarg;
                    LOG_INFO("Path to config file = %s \n", inputFile);
                    break;
            case 'i': 
                    inputimage = optarg;
                    LOG_INFO(" input image = %s \n",inputimage.c_str());
                    break;
            case 'o':
                    outputimage = optarg;
                    LOG_INFO(" output image = %s \n",outputimage.c_str());
                    break;

            default:
                LOG_INFO("Invalid parameter specified. Please run sample with the -h flag to see required arguments\n");
                exit(0);
        };
    }
    /**
     * To parse input,model and solution config from inputFile
    */
    Configuration::getInstance().LoadConfiguration(inputFile); 

    /**
     * To access enabled soultion model
    */
    vector<string> selected_model;
    /**
     * To access enabled solution configuration
    */
    vector<SolutionConfiguration> solutions_config;
    /**
     * To intialize each enabled solution
    */

    bool camera = false;
    for (auto i : Configuration::getInstance().solutionsconfig) { 
        /**
         * To access solution configuration
        */
         std::shared_ptr<SolutionConfiguration> config = i.second;
         /**
          * To check if solution is enabled
         */
         if (config->Enable == true) {
            /**
             * To access the input configuration 
            */
            config->input_config = Configuration::getInstance().inputconfigs[config->input_config_name];
            if (config->input_config == NULL) {
                LOG_ERROR("NULL Input configuration for selected solution name = %s \n", config->solution_name.c_str());
                exit(1);
            }
            config->input_config->StreamNumber = i.first;
            /**
             * To access the model configuration 
            */
            config->model_config = Configuration::getInstance().modelsconfig[config->model_name];
            if (config->model_config == NULL) {
                LOG_ERROR("NULL Model configuration for selected solution name = %s \n", config->solution_name.c_str());
                exit(1);
            }
            /**
             * To store the enabled solution configuration
            */
            solutions_config.emplace_back(*config);
            /**
             * Append the selected models
            */
            selected_model.push_back(config->model_name);

            if(config->input_config_name.compare("camera") == 0)
            {
                camera = true;
                const int MAX_QUEUE_DEPTH = 1;
                gDecodeQueue = make_shared<DecodeQueue>(MAX_QUEUE_DEPTH);
                encoderCtrl = make_shared<EncodeController>();
                captureCtrl = make_shared<CaptureController>();
                /**
                 * Intialize gstreamer pipeline to capture
                */
                captureCtrl->CreateCapture(config->input_config, gDecodeQueue);
                /**
                 * Intialze encoder to display or save frame
                */
                encoderCtrl->CreateEncoder(config);
            }
        }
    }
    /**
     * Check if any solution is enabled
    */
    if (selected_model.size() == 0) {
        LOG_ERROR("Solution not enabled, Enable the desired solution in config.json file\n");
        exit(1);
    }
    if(camera == true)
    {
        Inference_Camera((void *)(&solutions_config[0]) );
        gDecodeQueue->Unlock();
        captureCtrl->StopAll();
        captureCtrl->StopAll();
    }
    else
    {
        if(inputimage.empty() || outputimage.empty())
        {
            LOG_ERROR("Example: ai-solutions -c data/config.json -i image_path -o output_image\n");
            return 0;
        }
        Inference_Image((void *)(&solutions_config[0]), inputimage, outputimage );
    }


    return 0;
}
