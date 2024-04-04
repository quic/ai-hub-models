//==============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

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
//#include "QnnProfile.h"
//#include "QnnError.h"
//#include "QnnSystemContext.h"
#include "../include/QnnTypeMacros.hpp"
//#include "QnnTypes.h"
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

sample_app::QnnSampleApp::QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
                                       void* backendLibraryHandle,
                                       iotensor::OutputDataType outputDataType,
                                       iotensor::InputDataType inputDataType,
                                       sample_app::ProfilingLevel profilingLevel)
        : m_qnnFunctionPointers(qnnFunctionPointers),
          m_outputDataType(outputDataType),
          m_inputDataType(inputDataType),
          m_profilingLevel(profilingLevel),
          m_backendLibraryHandle(backendLibraryHandle),
          m_isBackendInitialized(false),
          m_isContextCreated(false) {
    return;
}

sample_app::QnnSampleApp::~QnnSampleApp() {
    // Free Profiling object if it was created
    if (nullptr != m_profileBackendHandle) {
//    LOGI("Freeing backend profile object.");
        if (QNN_PROFILE_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.profileFree(m_profileBackendHandle)) {
//      LOGE("Could not free backend profile handle.");
        }
    }
    // Free context if not already done
    if (m_isContextCreated) {
//    LOGI("Freeing context");
        if (QNN_CONTEXT_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
//      LOGE("Could not free context");
        }
    }
    m_isContextCreated = false;
    // Terminate backend
    if (m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) {
//    LOGI("Freeing backend");
        if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
//      LOGE("Could not free backend");
        }
    }
    m_isBackendInitialized = false;
    // Terminate logging in the backend
    if (nullptr != m_qnnFunctionPointers.qnnInterface.logFree && nullptr != m_logHandle) {
        if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logFree(m_logHandle)) {
//      LOGW("Unable to terminate logging in the backend.");
        }
    }
    qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    m_graphsInfo = nullptr;
    return;
}

std::string sample_app::QnnSampleApp::getBackendBuildId() {
    char* backendBuildId{nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
//    LOGE("Unable to get build Id from the backend.");
    }
    return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

// Initialize QnnSampleApp. Things it does:
//  1. Create output directory
//  2. Read all input list paths provided
//      during creation.
sample_app::StatusCode sample_app::QnnSampleApp::initialize() {
    // initialize logging in the backend
    if (log::isLogInitialized()) {
        auto logCallback = log::getLogCallback();
        auto logLevel    = log::getLogLevel();
//    LOGI("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
//             logCallback,
//             logLevel);
        if (QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
//      LOGI("Unable to initialize logging in the backend.");
        }
    } else {
//    LOGI("Logging not available in the backend.");
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::initializeProfiling() {
    if (ProfilingLevel::OFF != m_profilingLevel) {
//    LOGI("Profiling turned on; level = %d", m_profilingLevel);
        if (ProfilingLevel::BASIC == m_profilingLevel) {
//      LOGI("Basic profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR !=
                m_qnnFunctionPointers.qnnInterface.profileCreate(
                        m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
//        LOGI("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
//      LOGI("Detailed profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR !=
                m_qnnFunctionPointers.qnnInterface.profileCreate(
                        m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
//        LOGE("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
int32_t sample_app::QnnSampleApp::reportError(const std::string& err) {
//  LOGE("%s", err.c_str());
    return EXIT_FAILURE;
}

// Initialize a QnnBackend.
sample_app::StatusCode sample_app::QnnSampleApp::initializeBackend() {
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
            m_logHandle, (const QnnBackend_Config_t**)m_backendConfig, &m_backendHandle);
    if (QNN_BACKEND_NO_ERROR != qnnStatus) {
//    QNN_ERROR("Could not initialize backend due to error = %d", qnnStatus);
//    LOGE("Could not initialize backend due to error");
        return StatusCode::FAILURE;
    }
//  LOGI("Initialize Backend Returned Status success");
    m_isBackendInitialized = true;
    return StatusCode::SUCCESS;
}

// Terminate the backend after done.
sample_app::StatusCode sample_app::QnnSampleApp::terminateBackend() {
    if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
//    LOGE("Could not terminate backend");
        return StatusCode::FAILURE;
    }
    m_isBackendInitialized = false;
    return StatusCode::SUCCESS;
}

// Create a Context in a backend.
sample_app::StatusCode sample_app::QnnSampleApp::createContext() {
    if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(
            m_backendHandle,
            m_deviceHandle,
            (const QnnContext_Config_t**)&m_contextConfig,
            &m_context)) {
//    LOGE("Could not create context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = true;
    return StatusCode::SUCCESS;
}

// Free context after done.
sample_app::StatusCode sample_app::QnnSampleApp::freeContext() {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
//    LOGE("Could not free context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
    return StatusCode::SUCCESS;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in m_graphsInfo and m_graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
sample_app::StatusCode sample_app::QnnSampleApp::composeGraphs() {
    auto returnStatus = StatusCode::SUCCESS;
    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "composing\n");
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
        m_qnnFunctionPointers.composeGraphsFnHandle(
                m_backendHandle,
                m_qnnFunctionPointers.qnnInterface,
                m_context,
                (const qnn_wrapper_api::GraphConfigInfo_t**)m_graphConfigsInfo,
                m_graphConfigsInfoCount,
                &m_graphsInfo,
                &m_graphsCount,
                m_debug
                //log::getLogCallback(),
                //log::getLogLevel()
        )) {
//    LOGE("Failed in composeGraphs()");
        returnStatus = StatusCode::FAILURE;
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::finalizeGraphs() {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        if (QNN_GRAPH_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.graphFinalize(
                    (*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
            return StatusCode::FAILURE;
        }
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
    auto returnStatus = StatusCode::SUCCESS;
    if (!m_saveBinaryName.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Entering save binary\n");
        returnStatus = saveBinary();
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::createFromBinary(char* buffer, long bufferSize) {
    if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "QNN System function pointers are not populated\n");
        return StatusCode::FAILURE;
    }

    if (0 == bufferSize) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Received path to an empty file. Nothing to deserialize\n");
        return StatusCode::FAILURE;
    }
    if (!buffer) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Failed to allocate memory.\n");
        return StatusCode::FAILURE;
    }

    // inspect binary info
    auto returnStatus = StatusCode::SUCCESS;
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Could not create system handle.\n");
//    LOGE("Could not create system handle.");
        returnStatus = StatusCode::FAILURE;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle,
                static_cast<void*>(buffer),
                bufferSize,
                &binaryInfo,
                &binaryInfoSize)) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Failed to get context binary info\n");
//    LOGE("Failed to get context binary info");
        returnStatus = StatusCode::FAILURE;
    }

    // fill GraphInfo_t based on binary info
    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Failed to copy metadata.\n");
//    LOGE("Failed to copy metadata.");
        returnStatus = StatusCode::FAILURE;
    }
    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "contextCreateFromBinaryFnHandle is nullptr.\n");
//    LOGE("contextCreateFromBinaryFnHandle is nullptr.");
        returnStatus = StatusCode::FAILURE;
    }
    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                m_backendHandle,
                m_deviceHandle,
                (const QnnContext_Config_t**)&m_contextConfig,
                static_cast<void*>(buffer),
                bufferSize,
                &m_context,
                m_profileBackendHandle)) {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Could not create context from binary.\n");
//    LOGE("Could not create context from binary.");
        returnStatus = StatusCode::FAILURE;
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
    m_isContextCreated = true;
    if (StatusCode::SUCCESS == returnStatus) {
        for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
            if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
                __android_log_print(ANDROID_LOG_ERROR, "QNN ", "graphRetrieveFnHandle is nullptr.\n");
//        LOGE("graphRetrieveFnHandle is nullptr.");
                returnStatus = StatusCode::FAILURE;
                break;
            }
            if (QNN_SUCCESS !=
                m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                        m_context, (*m_graphsInfo)[graphIdx].graphName, &((*m_graphsInfo)[graphIdx].graph))) {
                __android_log_print(ANDROID_LOG_ERROR, "QNN ", "Unable to retrieve graph handle for graph Idx.\n");
//        LOGE("Unable to retrieve graph handle for graph Idx: %d", graphIdx);
                returnStatus = StatusCode::FAILURE;
            }
        }
    }
    if (StatusCode::SUCCESS != returnStatus) {
//    LOGI("Cleaning up graph Info structures.");
        qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::saveBinary() {
    if (m_saveBinaryName.empty()) {
//    LOGE("No name provided to save binary file.");
        return StatusCode::FAILURE;
    }
    if (nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinarySize ||
        nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinary) {
//    LOGE("contextGetBinarySizeFnHandle or contextGetBinaryFnHandle is nullptr.");
        return StatusCode::FAILURE;
    }
    uint64_t requiredBufferSize{0};
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context, &requiredBufferSize)) {
//    LOGE("Could not get the required binary size.");
        return StatusCode::FAILURE;
    }
    std::unique_ptr<uint8_t[]> saveBuffer(new uint8_t[requiredBufferSize]);
    if (nullptr == saveBuffer) {
//    LOGE("Could not allocate buffer to save binary.");
        return StatusCode::FAILURE;
    }
    uint64_t writtenBufferSize{0};
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context,
                                                            reinterpret_cast<void*>(saveBuffer.get()),
                                                            requiredBufferSize,
                                                            &writtenBufferSize)) {
//    LOGE("Could not get binary.");
        return StatusCode::FAILURE;
    }
    if (requiredBufferSize < writtenBufferSize) {
//    QNN_ERROR(
//        "Illegal written buffer size [%d] bytes. Cannot exceed allocated memory of [%d] bytes",
//        writtenBufferSize,
//        requiredBufferSize);
        return StatusCode::FAILURE;
    }
    auto dataUtilStatus = tools::datautil::writeBinaryToFile(
            m_outputPath, m_saveBinaryName + ".bin", (uint8_t*)saveBuffer.get(), writtenBufferSize);
    if (tools::datautil::StatusCode::SUCCESS != dataUtilStatus) {
//    LOGE("Error while writing binary to file.");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractBackendProfilingInfo(
        Qnn_ProfileHandle_t profileHandle) {
    if (nullptr == m_profileBackendHandle) {
//    LOGE("Backend Profile handle is nullptr; may not be initialized.");
        return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t* profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(
            profileHandle, &profileEvents, &numEvents)) {
//    LOGE("Failure in profile get events.");
        return StatusCode::FAILURE;
    }
//  LOGI("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
    for (size_t event = 0; event < numEvents; event++) {
        extractProfilingEvent(*(profileEvents + event));
        extractProfilingSubEvents(*(profileEvents + event));
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingSubEvents(
        QnnProfile_EventId_t profileEventId) {
    const QnnProfile_EventId_t* profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
            profileEventId, &profileSubEvents, &numSubEvents)) {
//    LOGE("Failure in profile get sub events.");
        return StatusCode::FAILURE;
    }
//  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
    for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
        extractProfilingEvent(*(profileSubEvents + subEvent));
        extractProfilingSubEvents(*(profileSubEvents + subEvent));
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingEvent(
        QnnProfile_EventId_t profileEventId) {
    QnnProfile_EventData_t eventData;
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
//    LOGE("Failure in profile get event type.");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
    auto returnStatus = sample_app::StatusCode::FAILURE;
    switch (errCode) {
        case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
            returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
            break;
        case QNN_COMMON_ERROR_SYSTEM:
            returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_ERROR;
            break;
        case QNN_COMMON_ERROR_NOT_SUPPORTED:
            returnStatus = sample_app::StatusCode::QNN_FEATURE_UNSUPPORTED;
            break;
        default:
            break;
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::isDevicePropertySupported() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnStatus =
                m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
//      LOGI("Device property is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
//      LOGI("Device property is not known to backend");
            return StatusCode::FAILURE;
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::createDevice() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceCreate) {
        auto qnnStatus =
                m_qnnFunctionPointers.qnnInterface.deviceCreate(m_logHandle, nullptr, &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
//      LOGE("Failed to create device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::freeDevice() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
        auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
//      LOGE("Failed to free device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode fillDims(std::vector<size_t>& dims,uint32_t* inDimensions, uint32_t rank) {
    if (nullptr == inDimensions) {
        QNN_ERROR("input dimensions is nullptr");
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "******************************************************************** input dimensions is nullptr\n");
        return sample_app::StatusCode::FAILURE;
    }
    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
    return sample_app::StatusCode::SUCCESS;
}

sample_app::StatusCode copyFromFloatToNative(float* floatBuffer,
                                             Qnn_Tensor_t* tensor) {
    if (nullptr == floatBuffer || nullptr == tensor) {
        QNN_ERROR("copyFromFloatToNative(): received a nullptr");
        return sample_app::StatusCode::FAILURE;
    }

    sample_app::StatusCode returnStatus = sample_app::StatusCode::SUCCESS;
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));

    for(int i = 0;i<dims.size();i++)
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "******************************************************************** input dims = %lu\n >>>>",dims[i]);
    }
    switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
        case QNN_DATATYPE_UFIXED_POINT_8:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","ufp8\n");
            datautil::floatToTfN<uint8_t>(static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                          floatBuffer,
                                          QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                          QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                          datautil::calculateElementCount(dims));
            break;

        case QNN_DATATYPE_UFIXED_POINT_16:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","ufp16\n");
            datautil::floatToTfN<uint16_t>(static_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                           floatBuffer,
                                           QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                           QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                           datautil::calculateElementCount(dims));
            break;

        case QNN_DATATYPE_UINT_8:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","uint8\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<uint8_t>(
                        static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<uint8_t>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_UINT_16:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","uint16\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<uint16_t>(
                        static_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<uint16_t>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_UINT_32:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","uint32\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<uint32_t>(
                        static_cast<uint32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<uint32_t>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_INT_8:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","int8\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<int8_t>(
                        static_cast<int8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<int8_t>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_INT_16:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","int16\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<int16_t>(
                        static_cast<int16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<int16_t>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_INT_32:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","int32\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<int32_t>(
                        static_cast<int32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<int32_t>");
                __android_log_print(ANDROID_LOG_ERROR, "QNN ","\"failure in castFromFloat<int32_t>\"\n");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        case QNN_DATATYPE_BOOL_8:
            __android_log_print(ANDROID_LOG_ERROR, "QNN ","bool8\n");
            if (datautil::StatusCode::SUCCESS !=
                datautil::castFromFloat<uint8_t>(
                        static_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                        floatBuffer,
                        datautil::calculateElementCount(dims))) {
                QNN_ERROR("failure in castFromFloat<bool>");
                returnStatus = sample_app::StatusCode::FAILURE;
            }
            break;

        default:
            QNN_ERROR("Datatype not supported yet!");
            __android_log_print(ANDROID_LOG_ERROR, "QNN ", "copyFromFloatToNative -> Datatype not supported yet!\n");
            returnStatus = sample_app::StatusCode::FAILURE;
            break;
    }
    return returnStatus;
}

// executeGraphs() that is currently used by qnn-sample-app's main.cpp.
// This function runs all the graphs present in model.so by reading
// inputs from input_list based files and writes output to .raw files.
sample_app::StatusCode sample_app::QnnSampleApp::executeGraphs(float* input_buffer,cv::Mat& output_buffer,std::vector<size_t> &output_dims) {
    auto returnStatus = StatusCode::SUCCESS;
    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "execute is running\n");
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        Qnn_Tensor_t* inputs  = nullptr;
        Qnn_Tensor_t* outputs = nullptr;
        if (iotensor::StatusCode::SUCCESS !=
            m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
            returnStatus = StatusCode::FAILURE;
            __android_log_print(ANDROID_LOG_ERROR, "QNN ", "input  fail\n");
            break;
        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "graphs = %d\n",m_graphsCount);
        auto graphInfo     = (*m_graphsInfo)[graphIdx];
        ///////////
        if (nullptr == inputs) {
            QNN_ERROR("inputs is nullptr");
            return StatusCode::FAILURE;
        }
        auto inputCount = graphInfo.numInputTensors;

        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "inputCount = %d\n",inputCount);
        for (size_t inputIdx = 0; inputIdx < inputCount; inputIdx++) {
            if (m_inputDataType == iotensor::InputDataType::FLOAT &&
                QNN_TENSOR_GET_DATA_TYPE(inputs) != QNN_DATATYPE_FLOAT_32) {
                float * fileToBuffer = input_buffer;
                for(int i=0;i<10;i++)
                {
                    __android_log_print(ANDROID_LOG_ERROR, "QNN ", " before buffer[%d] = %f\n",i,fileToBuffer[i]);
                }
                __android_log_print(ANDROID_LOG_ERROR, "QNN ", "##############################################\n");

                returnStatus = copyFromFloatToNative(reinterpret_cast<float *>(fileToBuffer), inputs); // Copy uint8_t* image to input
            }
            //TODO for multi input
            if (StatusCode::SUCCESS != returnStatus) {
//            LOGE("populateInputTensorFromFiles failed for input ");
                return returnStatus;
            }

        }
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "populateInputTensor done\n");

        float* bufferToWrite0 = reinterpret_cast<float*>(QNN_TENSOR_GET_CLIENT_BUF(outputs).data);
        for(int i=0;i<10;i++)
        {
            __android_log_print(ANDROID_LOG_ERROR, "QNN ", "before output buffer[%d] = %f\n",i,bufferToWrite0[i]);
        }
        ///////////
        if (StatusCode::SUCCESS == returnStatus) {
            Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
            executeStatus =
                    m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                    inputs,
                                                                    graphInfo.numInputTensors,
                                                                    outputs,
                                                                    graphInfo.numOutputTensors,
                                                                    m_profileBackendHandle,
                                                                    nullptr);
            if (QNN_GRAPH_NO_ERROR != executeStatus) {
                returnStatus = StatusCode::FAILURE;
            }
            if (StatusCode::SUCCESS == returnStatus) {
                //////////////////
                if (nullptr == outputs) {
//            LOGE("Received nullptr");
                    return StatusCode::FAILURE;
                }

                ///////////////////////TODO
//              returnStatus =returnStatus = convertAndWriteOutputTensorInFloat(
//          &(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);

                Qnn_Tensor_t* output = &outputs[0];
                fillDims(output_dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
                float* floatBuffer = nullptr;
//        returnStatus       = convertToFloat(&floatBuffer, output);
                size_t elementCount = datautil::calculateElementCount(output_dims);
//          returnStatus        = allocateBuffer<float>(&floatBuffer, elementCount);
                float** buffer = &floatBuffer;
                *buffer = (float*)malloc(elementCount * sizeof(float));
                if (nullptr == *buffer) {
                    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "mem alloc failed for *buffer\n");
                    return StatusCode::FAILURE;
                }
                if (StatusCode::SUCCESS != returnStatus) {
                    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "failure in allocateBuffer<float>\n");
                    return StatusCode::FAILURE;
                }
                Qnn_Tensor_t* tensor = output;
                float** out = &floatBuffer;
                switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
                    case QNN_DATATYPE_UFIXED_POINT_8:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::tfNToFloat<uint8_t>(
                                    *out,
                                    reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                    elementCount)) {
                            QNN_ERROR("failure in tfNToFloat<uint8_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_UFIXED_POINT_16:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::tfNToFloat<uint16_t>(
                                    *out,
                                    reinterpret_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                    elementCount)) {
                            QNN_ERROR("failure in tfNToFloat<uint8_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_UINT_8:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<uint8_t>(
                                    *out,
                                    reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<uint8_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_UINT_16:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<uint16_t>(
                                    *out,
                                    reinterpret_cast<uint16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<uint16_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_UINT_32:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<uint32_t>(
                                    *out,
                                    reinterpret_cast<uint32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<uint32_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_INT_8:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<int8_t>(
                                    *out,
                                    reinterpret_cast<int8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<int8_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_INT_16:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<int16_t>(
                                    *out,
                                    reinterpret_cast<int16_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<int16_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_INT_32:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<int32_t>(
                                    *out,
                                    reinterpret_cast<int32_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<int32_t>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    case QNN_DATATYPE_BOOL_8:
                        if (datautil::StatusCode::SUCCESS !=
                            datautil::castToFloat<uint8_t>(
                                    *out,
                                    reinterpret_cast<uint8_t*>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
                                    elementCount)) {
                            QNN_ERROR("failure in castToFloat<bool>");
                            returnStatus = StatusCode::FAILURE;
                        }
                        break;

                    default:
                        QNN_ERROR("Datatype not supported yet!");
                        returnStatus = StatusCode::FAILURE;
                        break;
                }
                if (StatusCode::SUCCESS != returnStatus) {
                    QNN_DEBUG("freeing *out");
                    if (*out != nullptr) {
                        free(*out);
                        *out = nullptr;
                    }
                }
                ///////////////////////////
                if (StatusCode::SUCCESS != returnStatus) {
                    QNN_ERROR("failure in convertToFloat");
                    return StatusCode::FAILURE;
                }
                float* bufferToWrite = reinterpret_cast<float*>(floatBuffer);
//            datautil::writeBatchDataToFile(
//                    outputPaths, fileName, dims, QNN_DATATYPE_FLOAT_32, bufferToWrite, outputBatchSize))

                //////////////////////////////
                size_t length = elementCount * sizeof(float);
                __android_log_print(ANDROID_LOG_ERROR, "QNN ", "after output length = %d\n",length);


                //////////////////////////////
                for(int i=0;i<10;i++)
                {
                    __android_log_print(ANDROID_LOG_ERROR, "QNN ", "after output buffer[%d] = %f\n",i,bufferToWrite[i]);
                }



                for (size_t outputIdx = 0; outputIdx < graphInfo.numOutputTensors; outputIdx++) {
                    std::string outputFilePrefix;
                    if (nullptr != QNN_TENSOR_GET_NAME(outputs[outputIdx]) &&
                        strlen(QNN_TENSOR_GET_NAME(outputs[outputIdx])) > 0) {
                        outputFilePrefix = std::string(QNN_TENSOR_GET_NAME(outputs[outputIdx]));
                    }


                    output_buffer  = cv::Mat(output_dims[1], output_dims[2], CV_32FC3, reinterpret_cast<uchar*>(bufferToWrite));

                }
                //////////////////

            }
        }
        if (StatusCode::SUCCESS != returnStatus) {
            break;
        }
//      }
//    }

        m_ioTensor.tearDownInputAndOutputTensors(
                inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
        inputs  = nullptr;
        outputs = nullptr;
        if (StatusCode::SUCCESS != returnStatus) {
            break;
        }
    }

    return returnStatus;
}