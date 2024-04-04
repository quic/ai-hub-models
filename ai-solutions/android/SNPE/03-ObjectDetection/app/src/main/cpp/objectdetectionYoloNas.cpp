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

#include "hpp/inference.h"
#include "hpp/Util.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"

#include "YOLONAS_Model.h"
#include "SSDMobileNetV2_Model.h"
#include "YOLO_X_Model.h"

Model *modelobj;

extern "C" JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1objdetect_SNPEHelper_queryRuntimes(
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
Java_com_qcom_aistack_1objdetect_SNPEHelper_initSNPE(JNIEnv *env, jobject thiz, jobject asset_manager, jchar runtime, jstring jdlc_name) {
    LOGI("Reading SNPE DLC ...");
    std::string result;

    //AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    //AAsset* asset_BB = AAssetManager_open(mgr, "Quant_yoloNas_s_320.dlc", AASSET_MODE_UNKNOWN);

    const char *cstr = env->GetStringUTFChars(jdlc_name, 0);
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset_BB = AAssetManager_open(mgr, cstr, AASSET_MODE_UNKNOWN);

    if(strcmp(cstr,"Quant_yoloNas_s_320.dlc")==0) {
        LOGI("Quant_yoloNas_s_320 dlc");
    }

    modelobj= new YOLONAS_Model();

    //Changing PrePost for different models
    if (strcmp(cstr,"ssd_mobilenetV2_without_ABP-NMS_Q.dlc")==0){
        LOGI("ssd_mobilenetV2_without_ABP-NMS_Q dlc");
        modelobj = new SSDMobileNetV2_Model();
        modelobj->msg();
    }
    else if(strcmp(cstr,"yolox_x_212_Q.dlc")==0){
        LOGI("YOLO_X dlc");
        modelobj = new YOLO_X_Model();
        modelobj->msg();
    }

    env->ReleaseStringUTFChars(jdlc_name, cstr);


    if (NULL == asset_BB) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }

    long dlc_size_BB = AAsset_getLength(asset_BB);
    LOGI("DLC BB Size = %ld MB\n", dlc_size_BB / (1024*1024));
    result += "DLC BB Size = " + std::to_string(dlc_size_BB);
    char* dlc_buffer_BB = (char*) malloc(sizeof(char) * dlc_size_BB);
    AAsset_read(asset_BB, dlc_buffer_BB, dlc_size_BB);

    result += "\n\nBuilding Models DLC Network:\n";
    result += build_network_BB(reinterpret_cast<const uint8_t *>(dlc_buffer_BB), dlc_size_BB,runtime, modelobj->model_name);

    return env->NewStringUTF(result.c_str());
}


//inference
extern "C"
JNIEXPORT jint JNICALL
Java_com_qcom_aistack_1objdetect_SNPEHelper_inferSNPE(JNIEnv *env, jobject thiz, jlong inputMat, jint actual_width, jint actual_height,
                                               jobjectArray jboxcoords, jobjectArray objnames) {

    LOGI("infer SNPE S");

    cv::Mat &img = *(cv::Mat*) inputMat;
    std::string bs;
    int numberofobj = 0;
    std::vector<std::vector<float>> BB_coords;
    std::vector<std::string> BB_names;

    bool status = executeDLC(img,actual_width, actual_height, numberofobj, BB_coords, BB_names, modelobj);

    if(numberofobj ==0)
        {
        LOGI("No object detected");
    }
    else if (numberofobj == -1){
        LOGE("ERROR in loading model properly");
        return -1;
    }
    else if(status == false)
    {
        LOGE("fatal ERROR");
        return 0;
    }
    else {
        //LOGI("number of detected objects: %d",numberofobj);

        for (int z = 0; z < numberofobj; z++){
            jfloatArray boxcoords = (jfloatArray) env->GetObjectArrayElement(jboxcoords, z);
            env->SetObjectArrayElement(objnames, z,env->NewStringUTF(BB_names[z].data()));


            float tempbox[5]; //4 coords and 1 processing time
            for(int k=0;k<5;k++)
                tempbox[k]=BB_coords[z][k];
            env->SetFloatArrayRegion(boxcoords,0,5,tempbox);
        }
        //LOGI("executeDLC_returned successfully");
    }
    //LOGD("infer SNPE E");
    return numberofobj;

}