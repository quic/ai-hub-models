//==============================================================================
//
//  Copyright (c) 2020, 2022, 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <memory>
#include <queue>

#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnSampleAppUtils.hpp"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"

namespace qnn {
namespace tools {
namespace iotensor {

enum class StatusCode { SUCCESS, FAILURE };
enum class OutputDataType { FLOAT_ONLY, NATIVE_ONLY, FLOAT_AND_NATIVE, INVALID };
enum class InputDataType { FLOAT, NATIVE, INVALID };

OutputDataType parseOutputDataType(std::string dataTypeString);
InputDataType parseInputDataType(std::string dataTypeString);

using PopulateInputTensorsRetType_t = std::tuple<StatusCode, size_t, size_t>;

class IOTensor {
 public:
  StatusCode setupInputAndOutputTensors(Qnn_Tensor_t **inputs,
                                        Qnn_Tensor_t **outputs,
                                        qnn_wrapper_api::GraphInfo_t graphInfo);

  StatusCode writeOutputTensors(uint32_t graphIdx,
                                size_t startIdx,
                                char *graphName,
                                Qnn_Tensor_t *outputs,
                                uint32_t numOutputs,
                                OutputDataType outputDatatype,
                                uint32_t graphsCount,
                                std::string outputPath,
                                size_t numInputFilesPopulated,
                                size_t outputBatchSize);

  PopulateInputTensorsRetType_t populateInputTensors(
      uint32_t graphIdx,
      const std::vector<std::vector<std::string>> &filePathsVector,
      const size_t filePathsIndexOffset,
      const bool loopBackToStart,
      const std::unordered_map<std::string, uint32_t> &inputNameToIndex,
      Qnn_Tensor_t *inputs,
      qnn_wrapper_api::GraphInfo_t graphInfo,
      iotensor::InputDataType inputDataType);

  StatusCode tearDownInputAndOutputTensors(Qnn_Tensor_t *inputs,
                                           Qnn_Tensor_t *outputs,
                                           size_t numInputTensors,
                                           size_t numOutputTensors);

 private:
  PopulateInputTensorsRetType_t populateInputTensor(const std::vector<std::string> &filePaths,
                                                    const size_t filePathsIndexOffset,
                                                    const bool loopBackToStart,
                                                    Qnn_Tensor_t *input,
                                                    InputDataType inputDataType);

  PopulateInputTensorsRetType_t readDataAndAllocateBuffer(const std::vector<std::string> &filePaths,
                                                          const size_t filePathsIndexOffset,
                                                          const bool loopBackToStart,
                                                          std::vector<size_t> dims,
                                                          Qnn_DataType_t dataType,
                                                          uint8_t **bufferToCopy);

  template <typename T>
  StatusCode allocateBuffer(T **buffer, size_t &elementCount);

  StatusCode convertToFloat(float **out, Qnn_Tensor_t *output);

  StatusCode convertAndWriteOutputTensorInFloat(Qnn_Tensor_t *output,
                                                std::vector<std::string> outputPaths,
                                                std::string fileName,
                                                size_t outputBatchSize);

  StatusCode writeOutputTensor(Qnn_Tensor_t *output,
                               std::vector<std::string> outputPaths,
                               std::string fileName,
                               size_t outputBatchSize);

  StatusCode allocateAndCopyBuffer(uint8_t **buffer, Qnn_Tensor_t *tensor);

  StatusCode tearDownTensors(Qnn_Tensor_t *tensors, uint32_t tensorCount);

  StatusCode allocateBuffer(uint8_t **buffer, std::vector<size_t> dims, Qnn_DataType_t dataType);

  StatusCode copyFromFloatToNative(float *floatBuffer, Qnn_Tensor_t *tensor);

  StatusCode setupTensors(Qnn_Tensor_t **tensors, uint32_t tensorCount, Qnn_Tensor_t *tensorsInfo);

  StatusCode fillDims(std::vector<size_t> &dims, uint32_t *inDimensions, uint32_t rank);
};
}  // namespace iotensor
}  // namespace tools
}  // namespace qnn