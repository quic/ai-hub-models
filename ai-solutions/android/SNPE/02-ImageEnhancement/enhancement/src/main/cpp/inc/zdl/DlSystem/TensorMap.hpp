//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/DlError.hpp"

#include "DlSystem/TensorMap.h"

namespace DlSystem {

class TensorMap : public Wrapper<TensorMap, Snpe_TensorMap_Handle_t, true> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_TensorMap_Delete};
public:

  TensorMap()
    : BaseType(Snpe_TensorMap_Create())
  {  }

  TensorMap(const TensorMap& other)
    : BaseType(Snpe_TensorMap_CreateCopy(other.handle()))
  {  }

  TensorMap(TensorMap&& other) noexcept
    : BaseType(std::move(other))
  {  }

  TensorMap& operator=(const TensorMap& other){
    if(this != &other){
      Snpe_TensorMap_Assign(other.handle(), handle());
    }
    return *this;
  }
  TensorMap& operator=(TensorMap&& other) noexcept{
    return moveAssign(std::move(other));
  }

  DlSystem::ErrorCode add(const char* name, ITensor* tensor){
    if(!tensor) return DlSystem::ErrorCode::SNPE_CAPI_BAD_ARGUMENT;
    Snpe_TensorMap_Add(handle(), name, getHandle(*tensor));
    return DlSystem::ErrorCode::NONE;
  }

  void remove(const char* name) noexcept{
    Snpe_TensorMap_Remove(handle(), name);
  }

  size_t size() const noexcept{
    return Snpe_TensorMap_Size(handle());
  }

  void clear() noexcept{
    Snpe_TensorMap_Clear(handle());
  }


  ITensor* getTensor(const char* name) const noexcept{
    return makeReference<ITensor>(Snpe_TensorMap_GetTensor_Ref(handle(), name));
  }

  StringList getTensorNames() const{
    return moveHandle(Snpe_TensorMap_GetTensorNames(handle()));
  }

};

} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, TensorMap)
