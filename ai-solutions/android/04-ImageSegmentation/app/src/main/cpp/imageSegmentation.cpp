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
using namespace cv;
#include <jni.h>
#include <string>
#include <iostream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <dlfcn.h>

#include "hpp/inference.h"
#include "hpp/Util.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"

#include <android/trace.h>


extern "C" JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1segmentation_SNPEHelper_queryRuntimes(
        JNIEnv* env,
        jobject /* this */,
        jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string runT_Status;
    std::string nativeLibPath = std::string(cstr);


    if (!SetAdspLibraryPath(nativeLibPath)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "Failed to set ADSP Library Path\n");

        runT_Status += "\nFailed to set ADSP Library Path\nTerminating";
        return env->NewStringUTF(runT_Status.c_str());
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
Java_com_qcom_aistack_1segmentation_SNPEHelper_initSNPE(JNIEnv *env, jobject thiz, jobject asset_manager, jchar runtime, jstring jdlc_name) {
    LOGI("Reading SNPE DLC ...");
    std::string result;
    const char *cstr = env->GetStringUTFChars(jdlc_name, 0);
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset = AAssetManager_open(mgr, cstr, AASSET_MODE_UNKNOWN);
    if (NULL == asset) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }

    long dlc_size= AAsset_getLength(asset);
    LOGI("DLC segmentation Size = %ld MB\n", dlc_size / (1024*1024));
    result += "DLC segmentation Size = " + std::to_string(dlc_size);
    char* dlc_buffer = (char*) malloc(sizeof(char) * dlc_size);
    AAsset_read(asset, dlc_buffer, dlc_size);

    result += "\n\nBuilding Models DLC Network:\n";
    result += build_network(reinterpret_cast<const uint8_t *>(dlc_buffer), dlc_size,runtime);

    return env->NewStringUTF(result.c_str());
}

//inference
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_qcom_aistack_1segmentation_SNPEHelper_inferSNPE(JNIEnv *env, jobject thiz, jlong jinputMat, jint actual_width, jint actual_height,
                                                        jfloatArray segment_pixels) {

    ATrace_beginSection("inferSNPE_native");

    cv::Mat &img = *(cv::Mat*) jinputMat;
    float inferenceTime = 0.00000f;
    std::vector<float32_t> segvector;

    cv::Mat resultmat;

    ATrace_beginSection("executeDLCNative");
    executeDLC(img, actual_width, actual_height, inferenceTime,resultmat);
    ATrace_endSection();
    try {
        ATrace_beginSection("writingdatatojava");
        env->SetFloatArrayRegion(segment_pixels, 0, actual_width * actual_height,
                                 reinterpret_cast<const float *>(resultmat.data));
        ATrace_endSection();
    }
    catch(...)
    {
        ATrace_endSection();
        LOGI("error caught...ignored");
    }

    ATrace_endSection();
    return inferenceTime;
}