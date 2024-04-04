//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <memory>


#include "Wrapper.hpp"
#include "SNPE.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/PlatformConfig.hpp"
#include "DlSystem/TensorShapeMap.hpp"

#include "DlSystem/DlEnums.hpp"

#include "DlSystem/IOBufferDataTypeMap.hpp"

#include "SNPE/SNPEBuilder.h"


namespace SNPE {

class SNPEBuilder : public Wrapper<SNPEBuilder, Snpe_SNPEBuilder_Handle_t> {
  friend BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_SNPEBuilder_Delete};
public:

  explicit SNPEBuilder(DlContainer::IDlContainer *container)
    :  BaseType(Snpe_SNPEBuilder_Create(getHandle(container)))
  {  }


  SNPEBuilder& setPerformanceProfile(DlSystem::PerformanceProfile_t performanceProfile){
    Snpe_SNPEBuilder_SetPerformanceProfile(handle(), static_cast<Snpe_PerformanceProfile_t>(performanceProfile));
    return *this;
  }

  SNPEBuilder& setProfilingLevel(DlSystem::ProfilingLevel_t profilingLevel){
    Snpe_SNPEBuilder_SetProfilingLevel(handle(), static_cast<Snpe_ProfilingLevel_t>(profilingLevel));
    return *this;
  }

  SNPEBuilder& setExecutionPriorityHint(DlSystem::ExecutionPriorityHint_t priority){
    Snpe_SNPEBuilder_SetExecutionPriorityHint(handle(), static_cast<Snpe_ExecutionPriorityHint_t>(priority));
    return *this;
  }

  SNPEBuilder& setOutputLayers(const DlSystem::StringList& outputLayerNames){
    Snpe_SNPEBuilder_SetOutputLayers(handle(), getHandle(outputLayerNames));
    return *this;
  }

  SNPEBuilder& setOutputTensors(const DlSystem::StringList& outputTensorNames){
    Snpe_SNPEBuilder_SetOutputTensors(handle(), getHandle(outputTensorNames));
    return *this;
  }

  SNPEBuilder& setUseUserSuppliedBuffers(int bufferMode){
    Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(handle(), bufferMode);
    return *this;
  }

  SNPEBuilder& setDebugMode(int debugMode){
    Snpe_SNPEBuilder_SetDebugMode(handle(), debugMode);
    return *this;
  }

  SNPEBuilder& setInputDimensions(const DlSystem::TensorShapeMap& inputDimensionsMap){
    Snpe_SNPEBuilder_SetInputDimensions(handle(), getHandle(inputDimensionsMap));
    return *this;
  }

  SNPEBuilder& setInitCacheMode(int cacheMode){
    Snpe_SNPEBuilder_SetInitCacheMode(handle(), cacheMode);
    return *this;
  }

  SNPEBuilder& setPlatformConfig(const DlSystem::PlatformConfig& platformConfigHandle){
    Snpe_SNPEBuilder_SetPlatformConfig(handle(), getHandle(platformConfigHandle));
    return *this;
  }

  SNPEBuilder& setRuntimeProcessorOrder(const DlSystem::RuntimeList& runtimeList){
    Snpe_SNPEBuilder_SetRuntimeProcessorOrder(handle(), getHandle(runtimeList));
    return *this;
  }

  SNPEBuilder& setUnconsumedTensorsAsOutputs(int setOutput){
    Snpe_SNPEBuilder_SetUnconsumedTensorsAsOutputs(handle(), setOutput);
    return *this;
  }

  SNPEBuilder& setTimeOut(uint64_t timeoutMicroSec){
    Snpe_SNPEBuilder_SetTimeOut(handle(), timeoutMicroSec);
    return *this;
  }


  SNPEBuilder& setBufferDataType(const DlSystem::IOBufferDataTypeMap& dataTypeMap){
    Snpe_SNPEBuilder_SetBufferDataType(handle(), getHandle(dataTypeMap));
    return *this;
  }

  SNPEBuilder& setSingleThreadedInit(int singleThreadedInit){
    Snpe_SNPEBuilder_SetSingleThreadedInit(handle(), singleThreadedInit);
    return *this;
  }

  SNPEBuilder& setCpuFixedPointMode(bool cpuFxpMode){
    Snpe_SNPEBuilder_SetCpuFixedPointMode(handle(), cpuFxpMode);
    return *this;
  }

  SNPEBuilder& setModelName(DlSystem::String modelName){
    Snpe_SNPEBuilder_SetModelName(handle(), modelName.c_str());
    return *this;
  }

  std::unique_ptr<SNPE> build() noexcept{
    auto h = Snpe_SNPEBuilder_Build(handle());
    return h ?  makeUnique<SNPE>(h) : nullptr;
  }

};

} // ns SNPE


ALIAS_IN_ZDL_NAMESPACE(SNPE, SNPEBuilder)
