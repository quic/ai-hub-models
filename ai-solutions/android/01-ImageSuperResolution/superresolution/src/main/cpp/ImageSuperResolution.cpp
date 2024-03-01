// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include <opencv2/core.hpp>
#include <jni.h>
#include <string>
#include <iostream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "hpp/inference.h"
#include "hpp/Util.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "ESRGAN.h"
#include "SESR.h"
#include "SRGAN.h"
#include "QuickSRNetLarge.h"
#include "QuickSRNetSmall.h"
#include "QuickSRNetMedium.h"
#include "XLSR.h"

#include <opencv2/imgproc/types_c.h>

using namespace cv;

Model *modelobj;

extern "C" JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1superres_SNPEHelper_queryRuntimes(
        JNIEnv* env,
        jobject /* this */,
        jstring native_dir_path) {

    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string runT_Status;
    std::string nativeLibPath = std::string(cstr);

//    runT_Status += "\nLibs Path : " + nativeLibPath + "\n";

    if (!SetAdspLibraryPath(nativeLibPath)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "Failed to set ADSP Library Path\n");

        runT_Status += "\nFailed to set ADSP Library Path\nTerminating";
        return env->NewStringUTF(runT_Status.c_str());
    }
    else
    {
        LOGI("ADSP found");
    }

    // ====================================================================================== //
    runT_Status = "Querying Runtimes : \n\n";
    // DSP unsignedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP,zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Absent\n");
        runT_Status += "UnsignedPD DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Present\n");
        runT_Status += "UnsignedPD DSP runtime : Present\n";
    }
    // DSP signedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Absent\n");
        runT_Status += "DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Present\n");
        runT_Status += "DSP runtime : Present\n";
    }
    // GPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Absent\n");
        runT_Status += "GPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Present\n");
        runT_Status += "GPU runtime : Present\n";
    }
    // CPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Absent\n");
        runT_Status += "CPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Present\n");
        runT_Status += "CPU runtime : Present\n";
    }

    return env->NewStringUTF(runT_Status.c_str());
}


//initializing network
extern "C"
JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1superres_SNPEHelper_initSNPE(JNIEnv *env, jobject thiz, jobject asset_manager, jchar runtime, jstring jdlc_name) {
    LOGI("Reading SNPE DLC ...");
    std::string result;

    const char *cstr = env->GetStringUTFChars(jdlc_name, 0);
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset_model = AAssetManager_open(mgr, cstr, AASSET_MODE_UNKNOWN);

    //Changing Preprocessing/PostProcessing for SESR
    if(strcmp(cstr,"sesr_quant_128_4.dlc")==0){
        modelobj = new SESR();
    }
    //Changing Preprocessing/PostProcessing for SRGAN
    else if(strcmp(cstr,"srgan_quant_128_4.dlc")==0){
        modelobj = new SRGAN();
    }
    //Changing Preprocessing/PostProcessing for ESRGAN
    else if(strcmp(cstr,"esrgan_quant_128_4.dlc")==0){
        modelobj = new ESRGAN();
    }
    //Changing Preprocessing/PostProcessing for XLSR
    else if(strcmp(cstr,"xlsr_quant_128_4.dlc")==0){
        modelobj = new XLSR();
    }
    //Changing Preprocessing/PostProcessing for Quick_SR_Large
    else if(strcmp(cstr,"quicksrnet_large_quant_128_4.dlc")==0){
        modelobj = new QuickSRNetLarge();
    }
    //Changing Preprocessing/PostProcessing for Quick_SR_medium
    else if(strcmp(cstr,"quicksrnet_medium_quant_128_4.dlc")==0){
        modelobj = new QuickSRNetMedium();
    }
    //Changing Preprocessing/PostProcessing for Quick_SR_Small
    else if(strcmp(cstr,"quicksrnet_small_quant_128_4.dlc")==0){
        modelobj = new QuickSRNetSmall();
    }
    else
    {
        LOGE("Model pre and post is not defined");
        return NULL;
    }

    modelobj->msg();
    env->ReleaseStringUTFChars(jdlc_name, cstr);

    if (NULL == asset_model) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }

    long dlc_size = AAsset_getLength(asset_model);
    LOGI("DLC Size = %ld MB\n", dlc_size / (1024*1024));
    result += "DLC Size = " + std::to_string(dlc_size);
    char* dlc_buffer = (char*) malloc(sizeof(char) * dlc_size);
    AAsset_read(asset_model, dlc_buffer, dlc_size);

    result += "\n\nBuilding Models DLC Network:\n";
    result += build_network(reinterpret_cast<const uint8_t *>(dlc_buffer), dlc_size,runtime);

    return env->NewStringUTF(result.c_str());
}

//inference
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_qcom_aistack_1superres_SNPEHelper_inferSNPE(JNIEnv *env, jobject thiz, jlong inputMat,
                                                 jlong outputMat) {

    LOGI("infer SNPE S");

    cv::Mat &inputimg = *(cv::Mat*) inputMat;
    cvtColor(inputimg,inputimg,CV_BGR2RGB);

    cv::Mat &outputimg = *(cv::Mat*) outputMat;

    float milli_time;

    bool status = executeDLC(inputimg, outputimg, milli_time, modelobj);

    if(status == false)
    {
        LOGE("fatal ERROR");
        return 0;
    }
    else {
        LOGI("status is TRUE");
        LOGI("rows: %d cols: %d",outputimg.rows,outputimg.cols);
    }
    LOGI("infer SNPE E");
    LOGI("milli_time: %f",milli_time);
    return milli_time;

}