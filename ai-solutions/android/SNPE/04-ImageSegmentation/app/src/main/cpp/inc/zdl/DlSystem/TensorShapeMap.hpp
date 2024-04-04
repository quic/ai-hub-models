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
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/DlError.hpp"

#include "DlSystem/TensorShapeMap.h"

namespace DlSystem {

class TensorShapeMap : public Wrapper<TensorShapeMap, Snpe_TensorShapeMap_Handle_t> {
  friend BaseType;
  using BaseType::BaseType;
  static constexpr DeleteFunctionType DeleteFunction{Snpe_TensorShapeMap_Delete};

public:
  TensorShapeMap()
    : BaseType(Snpe_TensorShapeMap_Create())
  {  }
  TensorShapeMap(const TensorShapeMap& other)
    : BaseType(Snpe_TensorShapeMap_CreateCopy(other.handle()))
  {  }
  TensorShapeMap(TensorShapeMap&& other) noexcept
    : BaseType(std::move(other))
  {  }

  TensorShapeMap& operator=(const TensorShapeMap& other){
    if(this != &other){
      Snpe_TensorShapeMap_Assign(other.handle(), handle());
    }
    return *this;
  }
  TensorShapeMap& operator=(TensorShapeMap&& other) noexcept{
    return moveAssign(std::move(other));
  }

  DlSystem::ErrorCode add(const char *name, const TensorShape& tensorShape){
    return static_cast<DlSystem::ErrorCode>(
      Snpe_TensorShapeMap_Add(handle(), name, getHandle(tensorShape))
    );
  }

  DlSystem::ErrorCode remove(const char* name) noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_TensorShapeMap_Remove(handle(), name));
  }

  size_t size() const noexcept{
    return Snpe_TensorShapeMap_Size(handle());
  }

  DlSystem::ErrorCode clear() noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_TensorShapeMap_Clear(handle()));
  }

  TensorShape getTensorShape(const char* name) const noexcept{
    return moveHandle(Snpe_TensorShapeMap_GetTensorShape(handle(), name));
  }

  StringList getTensorShapeNames() const{
    return moveHandle(Snpe_TensorShapeMap_GetTensorShapeNames(handle()));
  }

};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, TensorShapeMap)
