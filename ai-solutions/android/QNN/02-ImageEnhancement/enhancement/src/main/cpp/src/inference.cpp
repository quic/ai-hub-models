#include <jni.h>
#include <string>
#include <iostream>
#include <memory>
#include <thread>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "android/log.h"
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <android/trace.h>
#include <dlfcn.h>
#include <opencv2/gapi/core.hpp>

#include "BuildId.hpp"
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "../include/QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"
#include "../include/inference.h"
#include "../include/Model.h"
#include <dirent.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace  qnn::tools;
using namespace  qnn::tools::sample_app;
using namespace  qnn::tools::dynamicloadutil;
using namespace  qnn::tools::iotensor;

// Set to true, we will get more layers' outputs
sample_app::ProfilingLevel parsedProfilingLevel = sample_app::ProfilingLevel::OFF;
std::mutex mtx;
std::unique_ptr<sample_app::QnnSampleApp> app;
static void* sg_backendHandle{nullptr};
static void* sg_modelHandle{nullptr};

sample_app::StatusCode execStatus_thread;

std::string build_network(const char * modelPath_cstr, const char* backEndPath_cstr, char* buffer, long bufferSize)
{
    std::string modelPath(modelPath_cstr);
    std::string backEndPath(backEndPath_cstr);
    std::string outputLogger;
    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "model Lib Path = %s \n", modelPath_cstr);
    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "backend Lib Path = %s \n", backEndPath_cstr);

    QnnFunctionPointers qnnFunctionPointers;
    bool loadFromCachedBinary{std::strstr(backEndPath_cstr, "Htp") != NULL ||
                              std::strstr(backEndPath_cstr, "Dsp") != NULL};
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                              modelPath,
                                                              &qnnFunctionPointers,
                                                              &sg_backendHandle,
                                                              !loadFromCachedBinary,
                                                              &sg_modelHandle);

    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "getQnnFunctionPointers done\n");
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
            outputLogger = "Error initializing QNN Function Pointers: could not load backend: " + backEndPath;
//            LOGE(outputLogger);
            return outputLogger;
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            outputLogger = "Error initializing QNN Function Pointers: could not load model: " + modelPath;
//            LOGE(outputLogger);
            return outputLogger;
        } else {
            outputLogger = "Error initializing QNN Function Pointers";
//            LOGE(outputLogger);
            return outputLogger;
        }
    }

    iotensor::OutputDataType parsedOutputDataType   = iotensor::OutputDataType::FLOAT_ONLY;
    iotensor::InputDataType parsedInputDataType     = iotensor::InputDataType::FLOAT;

    if (loadFromCachedBinary) {
        statusCode =
                dynamicloadutil::getQnnSystemFunctionPointers("libQnnSystem.so", &qnnFunctionPointers);
        if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
            exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
        }
    }

    app.reset(new sample_app::QnnSampleApp(qnnFunctionPointers,
                                       sg_backendHandle,
                                       parsedOutputDataType,
                                       parsedInputDataType,
                                       parsedProfilingLevel));

    if (sample_app::StatusCode::SUCCESS != app->initialize()) {
        outputLogger = "Initialization failure";
//        LOGE(outputLogger);
        return outputLogger;
    }


    if (sample_app::StatusCode::SUCCESS != app->initializeBackend()) {
        outputLogger = "Backend Initialization failure";
//        LOGE(outputLogger);
        return outputLogger;
    }

    auto devicePropertySupportStatus = app->isDevicePropertySupported();
    if (sample_app::StatusCode::FAILURE != devicePropertySupportStatus) {
        auto createDeviceStatus = app->createDevice();
        if (sample_app::StatusCode::SUCCESS != createDeviceStatus) {
            outputLogger = "Device Creation failure";
//            LOGE(outputLogger);
            return outputLogger;
        }
    }

    if (sample_app::StatusCode::SUCCESS != app->initializeProfiling()) {
        outputLogger = "Profiling Initialization failure";
//        LOGE(outputLogger);
        return outputLogger;
    }

    if (!loadFromCachedBinary) {
        if (sample_app::StatusCode::SUCCESS != app->createContext()) {
            outputLogger = "Context Creation failure";
//            LOGE(outputLogger);
            return outputLogger;
        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "createContext done\n");

        if (sample_app::StatusCode::SUCCESS != app->composeGraphs()) {
            outputLogger = "Graph Prepare failure";
//            LOGE(outputLogger);
            return outputLogger;
        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "composeGraphs done\n");

        if (sample_app::StatusCode::SUCCESS != app->finalizeGraphs()) {
            outputLogger = "Graph Finalize failure";
//            LOGE(outputLogger);
            return outputLogger;
        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "finalizeGraphs done\n");

    } else {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "create binary\n");
        if (sample_app::StatusCode::SUCCESS != app->createFromBinary(buffer, bufferSize)) {
            outputLogger = "Create From Binary failure";
//            LOGE(outputLogger);
            return outputLogger;
        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "else.............\n");
    }

    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "sample app done\n");
    outputLogger = " success ";
    return outputLogger;

    //TODO

}

bool executeModel(cv::Mat &inputimg, cv::Mat &outputimg, float &milli_time, Model *modelobj) {

    LOGI("execute_MODEL");
    ATrace_beginSection("preprocessing");
    std::vector<int> dims_in;
    LOGI("before pre: input image SIZE width::%d height::%d", inputimg.cols, inputimg.rows);

    dims_in.push_back(480);
    dims_in.push_back(640);
    dims_in.push_back(3);
    modelobj->preprocess(inputimg,dims_in);

    struct timeval start_time, end_time;
    float seconds, useconds;

    mtx.lock();
    assert(app != nullptr);

    ATrace_endSection();
    gettimeofday(&start_time, NULL);
    ATrace_beginSection("inference time");

    LOGI("shubham waiting");
    std::vector<size_t> dims;
    LOGI("width = %d",inputimg.cols);
    LOGI("height = %d\n",inputimg.rows);
    LOGI("channel = %d\n",inputimg.channels());

    cv::Mat out;
    execStatus_thread  = app->executeGraphs(reinterpret_cast<float *>(inputimg.data),out,dims, (modelobj->model_name == "RUAS") ? 6 : 0);
    sample_app::StatusCode execStatus = execStatus_thread;
    ATrace_endSection();
    ATrace_beginSection("postprocessing time");
    gettimeofday(&end_time, NULL);
    seconds = end_time.tv_sec - start_time.tv_sec; //seconds
    useconds = end_time.tv_usec - start_time.tv_usec; //milliseconds
    milli_time = ((seconds) * 1000 + useconds/1000.0);
    LOGI("Inference time %f ms", milli_time);

    if(execStatus== sample_app::StatusCode::SUCCESS){
        LOGI("Exec status is true");
    }
    else{
        LOGE("Exec status is false");
        mtx.unlock();
        return false;
    }

    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "---------------post=---------------\n");

    for(int i = 0;i<dims.size();i++)
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "********************************************************************dims = %lu\n >>>>",dims[i]);
    }


    std::vector<float32_t> databuffer;
    float32_t* buffer_data = reinterpret_cast<float32_t *>(out.data);
    int buffer_length = out.cols * out.rows * out.channels();
    for(int i=0;i<buffer_length;i++){
        databuffer.push_back(buffer_data[i]);
    }
    outputimg = cv::Mat(dims[1], dims[2], CV_32FC3, databuffer.data(), cv::Mat::AUTO_STEP);

    struct timeval pp_start_time, pp_end_time;
    float pp_seconds, pp_useconds, pp_milli_time;
    gettimeofday(&pp_start_time, NULL);

    modelobj->postprocess(outputimg);

    gettimeofday(&pp_end_time, NULL);
    pp_seconds = pp_end_time.tv_sec - pp_start_time.tv_sec; //seconds
    pp_useconds = pp_end_time.tv_usec - pp_start_time.tv_usec; //milliseconds
    pp_milli_time = ((pp_seconds) * 1000 + pp_useconds/1000.0);
    LOGI("Post processing time %f ms", pp_milli_time);

    ATrace_endSection();
    mtx.unlock();
    return true;
}

