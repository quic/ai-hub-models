// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "hpp/inference.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_qa_ml_QaClient_queryRuntimes(JNIEnv *env,
                                                   jobject thiz,
                                                   jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, NULL);
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


extern "C"
JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_qa_ml_QaClient_initSNPE(JNIEnv *env,
                                              jobject thiz,
                                              jobject asset_manager, jstring model) {
    LOGI("Reading SNPE DLC ...");
    std::string result;

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);

    //Converting the model name to string
    const char *accl_name = env->GetStringUTFChars(model, NULL);
    //AAsset* asset = AAssetManager_open(mgr, "alberta_int.dlc", AASSET_MODE_UNKNOWN);
    AAsset* asset;
    if(*accl_name == 'alberta'){
        asset = AAssetManager_open(mgr, "alberta_float.dlc", AASSET_MODE_UNKNOWN);
    }
    else if(*accl_name=='mobile_bert'){
        asset = AAssetManager_open(mgr, "mobile_bert_float.dlc", AASSET_MODE_UNKNOWN);
    }
    else{
        asset = AAssetManager_open(mgr, "electra_small_squad2_cached.dlc", AASSET_MODE_UNKNOWN);
    }

    if (NULL == asset) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }
    long dlc_size = AAsset_getLength(asset);

    char* dlc_buffer = (char*) malloc(sizeof(char) * dlc_size);
    AAsset_read(asset, dlc_buffer, dlc_size);

    result += build_network(reinterpret_cast<const uint8_t *>(dlc_buffer), dlc_size);

    return env->NewStringUTF(result.c_str());
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_qualcomm_qti_qa_ml_QaClient_inferSNPE(JNIEnv *env, jobject thiz, jstring runtime,
                                               jstring model,
                                               jfloatArray input_ids, jfloatArray attn_masks,
                                               jfloatArray seg_ids,
                                               jint array_size,
                                               jfloatArray start_logits,
                                               jfloatArray end_logits) {
    std::string return_msg;
    jfloat * inp_id_array;
    jfloat * mask_array;
    jfloat * tty_array;
    jint arrayLength = array_size;
    jfloat * sLogit_array;
    jfloat * eLogit_array;

    const char *accl_name = env->GetStringUTFChars(runtime, NULL);
    env->ReleaseStringUTFChars(runtime, accl_name);
    std::string backend = std::string(accl_name);

    // get a pointer to the array
    inp_id_array = env->GetFloatArrayElements(input_ids, NULL);
    mask_array = env->GetFloatArrayElements(attn_masks, NULL);
    tty_array = env->GetFloatArrayElements(seg_ids, NULL);
    sLogit_array = env->GetFloatArrayElements(start_logits, NULL);
    eLogit_array = env->GetFloatArrayElements(end_logits, NULL);

    // do some exception checking
    if (inp_id_array == NULL || mask_array == NULL) {
        return_msg += "0.0 0.0 Err: Invalid input_id_array/attn_mask_arr/arr_sizes\n";
        return env->NewStringUTF(return_msg.c_str()); /* exception occurred */
    }

    std::vector<float *> inputVec { inp_id_array, mask_array, tty_array };
    std::vector<float *> outputVec {eLogit_array, sLogit_array};

    return_msg = execute_net(inputVec, arrayLength, outputVec, backend);

    for ( int index = 0; index < arrayLength; index++ ) {
//        LOGI("out[%d] = %f ", index, outputVec[1][index]);
        eLogit_array[index] = outputVec[0][index];
        sLogit_array[index] = outputVec[1][index];
    }

    // ===================================================================== //
    // release the memory so java can have it again
    env->ReleaseFloatArrayElements(input_ids, inp_id_array, 0);
    env->ReleaseFloatArrayElements(attn_masks, mask_array, 0);
    env->ReleaseFloatArrayElements(seg_ids, tty_array, 0);

    env->ReleaseFloatArrayElements(start_logits, sLogit_array, 0);
    env->ReleaseFloatArrayElements(end_logits, eLogit_array, 0);

    return env->NewStringUTF(return_msg.c_str());
}