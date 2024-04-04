#include <opencv2/core.hpp>
#include <jni.h>
#include <string>
#include <iostream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

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

bool SetAdspLibraryPath(std::string nativeLibPath) {
    nativeLibPath += ";/data/local/tmp/mv_dlc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";

    __android_log_print(ANDROID_LOG_INFO, "QNN ", "ADSP Lib Path = %s \n", nativeLibPath.c_str());
    std::cout << "ADSP Lib Path = " << nativeLibPath << std::endl;

    return setenv("ADSP_LIBRARY_PATH", nativeLibPath.c_str(), 1 /*override*/) == 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qcom_aistack_1segmentation_QNNHelper_queryRuntimes(
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
Java_com_qcom_aistack_1segmentation_QNNHelper_initQNN(JNIEnv *env, jobject thiz, jobject asset_manager, jstring backend, jstring jmodel_name,jstring nativeDirPath) {
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

    const char *cstr_model_name =  env->GetStringUTFChars(jmodel_name , 0);
    const char *cstr_backend_name =  env->GetStringUTFChars(backend, 0);

    std::string model_string =  std::string(cstr_model_name);
    std::string backend_string = std::string(cstr_backend_name);

    long bin_size = AAsset_getLength(asset_model);
    LOGI("BIN Size = %ld KB\n", bin_size / (1024));
    result += "BIN Size = " + std::to_string(bin_size);
    char* bin_buffer = (char*) malloc(sizeof(char) * bin_size);
    AAsset_read(asset_model, bin_buffer, bin_size);

    result += "\n\nsuccess Building Models Network:\n";
    result += build_network(model_string.c_str(),backend_string.c_str(), bin_buffer, bin_size);


    return env->NewStringUTF(result.c_str());
}

//inference
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_qcom_aistack_1segmentation_QNNHelper_inferQNN(JNIEnv *env, jobject thiz, jlong jinputMat, jint actual_width, jint actual_height,
                                                       jfloatArray segment_pixels) {
    LOGI("infer QNN S");

    cv::Mat &inputimg = *(cv::Mat*) jinputMat;
    cvtColor(inputimg,inputimg,CV_BGR2RGB);

    float milli_time = 0.0;
    std::vector<float32_t> segvector;
    cv::Mat resultmat;

    bool status = executeModel(inputimg, actual_width, actual_height, milli_time, resultmat);

    if(status == false)
    {
        LOGE("fatal ERROR");
        return 0;
    }
    else {
        env->SetFloatArrayRegion(segment_pixels, 0, actual_width * actual_height,
                                 reinterpret_cast<const float *>(resultmat.data));
        LOGI("status is TRUE");
    }
    LOGI("infer QNN E");
    LOGI("milli_time: %f",milli_time);
    return milli_time;

}