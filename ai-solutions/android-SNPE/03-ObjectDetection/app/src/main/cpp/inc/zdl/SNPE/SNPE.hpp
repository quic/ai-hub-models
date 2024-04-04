//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UserMemoryMap.hpp"
#include "DlSystem/IBufferAttributes.hpp"
#include "DiagLog/IDiagLog.hpp"

#include "DlSystem/DlOptional.hpp"


#include "SNPE/SNPE.h"

namespace SNPE{

class SNPE : public Wrapper<SNPE, Snpe_SNPE_Handle_t, true> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_SNPE_Delete};

  template<typename T, typename H>
  static DlSystem::Optional<T> makeOptional(H handle){
    return DlSystem::Optional<T>(T(moveHandle(handle)));
  }
public:

  DlSystem::Optional<DlSystem::StringList> getInputTensorNames() const noexcept{
    return makeOptional<DlSystem::StringList>(Snpe_SNPE_GetInputTensorNames(handle()));
  }

  DlSystem::Optional<DlSystem::StringList> getOutputTensorNames() const noexcept{
    return makeOptional<DlSystem::StringList>(Snpe_SNPE_GetOutputTensorNames(handle()));
  }

  DlSystem::StringList getOutputTensorNamesByLayerName(const char *name) const noexcept{
    return moveHandle(Snpe_SNPE_GetOutputTensorNamesByLayerName(handle(), name));
  }

  bool execute(const DlSystem::TensorMap& input, DlSystem::TensorMap& output) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_ExecuteITensors(handle(), getHandle(input), getHandle(output));
  }


  bool execute(const DlSystem::ITensor* input, DlSystem::TensorMap& output) noexcept{
    if(!input) return false;
    return SNPE_SUCCESS == Snpe_SNPE_ExecuteITensor(handle(), getHandle(*input), getHandle(output));
  }

  bool execute(const DlSystem::UserBufferMap& input, const DlSystem::UserBufferMap& output) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_ExecuteUserBuffers(handle(), getHandle(input), getHandle(output));
  }


  /* To be deprecated, please use new api registerMemoryMappedBuffers */
  bool registerIonBuffers(const DlSystem::UserMemoryMap& ionBufferMap) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_RegisterUserMemoryMappedBuffers(handle(), getHandle(ionBufferMap));
  }

  /* To be deprecated, please use new api deregisterMemoryMappedBuffers */
  bool deregisterIonBuffers(const DlSystem::StringList& ionBufferNames) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_DeregisterUserMemoryMappedBuffers(handle(), getHandle(ionBufferNames));
  }

  bool registerMemoryMappedBuffers(const DlSystem::UserMemoryMap& memoryMappedBufferMap) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_RegisterUserMemoryMappedBuffers(handle(), getHandle(memoryMappedBufferMap));
  }

  bool deregisterMemoryMappedBuffers(const DlSystem::StringList& bufferNames) noexcept{
    return SNPE_SUCCESS == Snpe_SNPE_DeregisterUserMemoryMappedBuffers(handle(), getHandle(bufferNames));
  }

  std::string getModelVersion() const{
    auto str = Snpe_SNPE_GetModelVersion(handle());
    return str ? str : "";
  }

  DlSystem::Optional<DlSystem::TensorShape> getInputDimensions() const noexcept{
    return makeOptional<DlSystem::TensorShape>(Snpe_SNPE_GetInputDimensionsOfFirstTensor(handle()));
  }

  DlSystem::Optional<DlSystem::TensorShape> getInputDimensions(const char* name) const noexcept{
    return makeOptional<DlSystem::TensorShape>(Snpe_SNPE_GetInputDimensions(handle(), name));
  }

  DlSystem::Optional<DlSystem::StringList> getOutputLayerNames() const noexcept{
    return makeOptional<DlSystem::StringList>(Snpe_SNPE_GetOutputLayerNames(handle()));
  }


  DlSystem::Optional<DlSystem::IBufferAttributes*> getInputOutputBufferAttributes(const char* name) const noexcept{
    return DlSystem::Optional<DlSystem::IBufferAttributes*>(
      new DlSystem::IBufferAttributes(moveHandle(Snpe_SNPE_GetInputOutputBufferAttributes(handle(), name))),
      DlSystem::Optional<DlSystem::IBufferAttributes*>::LIFECYCLE::POINTER_OWNED
    );
  }

  DlSystem::Optional<DiagLog::IDiagLog*> getDiagLogInterface() noexcept{
    auto diagLogHandle = Snpe_SNPE_GetDiagLogInterface_Ref(handle());
    if(!diagLogHandle) return {};
    // Bind lifespan of this reference to this object
    auto toret = makeReference<DiagLog::IDiagLog>(diagLogHandle);
    return {toret, DlSystem::Optional<DiagLog::IDiagLog*>::LIFECYCLE::POINTER_NOT_OWNED};
  }

private:
  SNPE(const SNPE&) = delete;
  SNPE& operator=(const SNPE&) = delete;

};

} // ns SNPE

ALIAS_IN_ZDL_NAMESPACE(SNPE, SNPE)
