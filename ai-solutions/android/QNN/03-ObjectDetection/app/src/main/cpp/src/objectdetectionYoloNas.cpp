#include <opencv2/core.hpp>
#include <jni.h>
#include <string>
#include <iostream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "../include/YOLONAS_Model.h"
#include "../include/SSDMobileNetV2_Model.h"
#include "../include/YOLO_X_Model.h"
#include <opencv2/imgproc/types_c.h>
#include "../include/inference.h"

#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include "android/log.h"

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "../include/QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"
#include "../include/QnnWrapperUtils.hpp"
#include "../include/QnnTypeMacros.hpp"
#include "IOTensor.hpp"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/gapi/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace qnn;
using namespace qnn::tools;
using namespace qnn_wrapper_api;

using namespace qnn::tools;
using namespace cv;

Model *modelobj;

bool SetAdspLibraryPath(std::string nativeLibPath) {
    nativeLibPath += ";/data/local/tmp/mv_dlc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";

    __android_log_print(ANDROID_LOG_INFO, "QNN ", "ADSP Lib Path = %s \n", nativeLibPath.c_str());
    std::cout << "ADSP Lib Path = " << nativeLibPath << std::endl;

    return setenv("ADSP_LIBRARY_PATH", nativeLibPath.c_str(), 1 /*override*/) == 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1objdetect_QNNHelper_queryRuntimes(
        JNIEnv* env,
        jobject /* this */,
        jstring native_dir_path) {

    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string runT_Status;
    std::string nativeLibPath = std::string(cstr);

//    runT_Status += "\nLibs Path : " + nativeLibPath + "\n";

    if (!SetAdspLibraryPath(nativeLibPath)) {
        __android_log_print(ANDROID_LOG_INFO, "QNN ", "Failed to set ADSP Library Path\n");

        runT_Status += "\nFailed to set ADSP Library Path\nTerminating";
        return env->NewStringUTF(runT_Status.c_str());
    }
    else
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "native_dir_path = %s\n",cstr);
        LOGI("ADSP found");
    }

    // ====================================================================================== //
    runT_Status = "Querying Runtimes : \n\n";
    return env->NewStringUTF(runT_Status.c_str());
}


//initializing network
extern "C"
JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1objdetect_QNNHelper_initQNN(JNIEnv *env, jobject thiz, jobject asset_manager, jstring backend, jstring jmodel_name,jstring nativeDirPath) {
    LOGI("Reading QNN binary ...");
    std::string result;

    const char *cstr = env->GetStringUTFChars(jmodel_name, 0);
    const char *cstr_backend = env->GetStringUTFChars(backend, 0);

    const char *cstr_nativeDirPath = env->GetStringUTFChars(nativeDirPath, nullptr);
    env->ReleaseStringUTFChars(nativeDirPath, cstr_nativeDirPath);

    std::string nativeLibPath = std::string(cstr);
    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "native_dir_path = %s\n",nativeLibPath.c_str());

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset_model = AAssetManager_open(mgr, cstr, AASSET_MODE_UNKNOWN);
    AAsset* asset_lib = AAssetManager_open(mgr, cstr_backend, AASSET_MODE_UNKNOWN);

    //Changing PrePost for different models
    if(strcmp(cstr,"yolox_a8w8_2_15_1.serialized.bin")==0 || strcmp(cstr,"libyolox_a8w8_2_15_1.so")==0){
        modelobj = new YOLO_X_Model();
    }
    else if(strcmp(cstr,"yolo_nas_w8a8.serialized.bin")==0 || strcmp(cstr,"libyolo_nas_w8a8.so")==0){
        modelobj = new YOLONAS_Model();
    }
    else if(strcmp(cstr,"ssd_mobilenetV2_without_ABP-NMS_a8w8.serialized.bin")==0 || strcmp(cstr,"libssd_mobilenetV2_without_ABP-NMS_a8w8.so")==0){
        modelobj = new SSDMobileNetV2_Model();
    }
    else
    {
        LOGE("Model pre and post is not defined");
        return NULL;
    }

    const char *cstr_model_name =  env->GetStringUTFChars(jmodel_name , 0);
    const char *cstr_backend_name =  env->GetStringUTFChars(backend, 0);

    std::string model_string =  std::string(cstr_model_name);
    std::string backend_string = std::string(cstr_backend_name);

    long bin_size = AAsset_getLength(asset_model);
    LOGI("BIN Size = %ld KB\n", bin_size / (1024));
    result += "BIN Size = " + std::to_string(bin_size);
    char* bin_buffer = (char*) malloc(sizeof(char) * bin_size);
    AAsset_read(asset_model, bin_buffer, bin_size);

    result += "\n\nBuilding Models Network:\n";
    result += build_network(model_string.c_str(),backend_string.c_str(), bin_buffer, bin_size);
    result += " success ";

    return env->NewStringUTF(result.c_str());
}

//inference
extern "C"
JNIEXPORT jint JNICALL
Java_com_qcom_aistack_1objdetect_QNNHelper_inferQNN(JNIEnv *env, jobject thiz, jlong inputMat, jint actual_width, jint actual_height,
                                                   jobjectArray jboxcoords, jobjectArray objnames) {

    LOGI("infer QNN S");

    cv::Mat &img = *(cv::Mat*) inputMat;
    std::string bs;
    int numberofobj = 0;
    std::vector<std::vector<float>> BB_coords;
    std::vector<std::string> BB_names;

    bool status = executeModel(img,actual_width, actual_height, numberofobj, BB_coords, BB_names, modelobj);

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
        LOGI("number of detected objects: %d",numberofobj);

        for (int z = 0; z < numberofobj; z++){
            jfloatArray boxcoords = (jfloatArray) env->GetObjectArrayElement(jboxcoords, z);
            env->SetObjectArrayElement(objnames, z,env->NewStringUTF(BB_names[z].data()));


            float tempbox[5]; //4 coords and 1 processing time
            for(int k=0;k<5;k++)
                tempbox[k]=BB_coords[z][k];
            env->SetFloatArrayRegion(boxcoords,0,5,tempbox);
        }
    }
    LOGI("returning numberofobj: %d",numberofobj);
    return numberofobj;

}