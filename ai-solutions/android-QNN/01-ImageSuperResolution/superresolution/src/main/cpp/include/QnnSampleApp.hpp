//==============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <memory>
#include <queue>

#include "IOTensor.hpp"
#include "SampleApp.hpp"
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>


namespace qnn {
namespace tools {
namespace sample_app {

enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

class QnnSampleApp {
 public:
  QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
               void *backendHandle,
               iotensor::OutputDataType outputDataType = iotensor::OutputDataType::FLOAT_ONLY,
               iotensor::InputDataType inputDataType   = iotensor::InputDataType::FLOAT,
               ProfilingLevel profilingLevel           = ProfilingLevel::OFF);

  // @brief Print a message to STDERR then return a nonzero
  //  exit status.
  int32_t reportError(const std::string &err);

  StatusCode initialize();

  StatusCode initializeBackend();

  StatusCode createContext();

  StatusCode composeGraphs();

  StatusCode finalizeGraphs();

  StatusCode executeGraphs(float* input_buffer,cv::Mat& output_buffer,std::vector<size_t> &output_dims);

  StatusCode registerOpPackages();

  StatusCode createFromBinary(char* buffer, long bufferSize);

  StatusCode saveBinary();

  StatusCode freeContext();

  StatusCode terminateBackend();

  StatusCode freeGraphs();

  Qnn_ContextHandle_t getContext();

  StatusCode initializeProfiling();

  std::string getBackendBuildId();

  StatusCode isDevicePropertySupported();

  StatusCode createDevice();

  StatusCode freeDevice();

  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  virtual ~QnnSampleApp();

 private:
  StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

  static const std::string s_defaultOutputPath;

  QnnFunctionPointers m_qnnFunctionPointers;
  std::vector<std::string> m_inputListPaths;
  std::vector<std::vector<std::vector<std::string>>> m_inputFileLists;
  std::vector<std::unordered_map<std::string, uint32_t>> m_inputNameToIndex;
  std::vector<std::string> m_opPackagePaths;
  std::string m_outputPath;
  std::string m_saveBinaryName;
  std::string m_cachedBinaryPath;
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_ContextHandle_t m_context         = nullptr;
  QnnContext_Config_t **m_contextConfig = nullptr;
  bool m_debug;
  iotensor::OutputDataType m_outputDataType;
  iotensor::InputDataType m_inputDataType;
  ProfilingLevel m_profilingLevel;
  bool m_dumpOutputs;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
  uint32_t m_graphsCount;
  void *m_backendLibraryHandle;
  iotensor::IOTensor m_ioTensor;
  bool m_isBackendInitialized;
  bool m_isContextCreated;
  Qnn_ProfileHandle_t m_profileBackendHandle              = nullptr;
  qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
  uint32_t m_graphConfigsInfoCount;
  Qnn_LogHandle_t m_logHandle         = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle   = nullptr;
};
}  // namespace sample_app
}  // namespace tools
}  // namespace qnn
