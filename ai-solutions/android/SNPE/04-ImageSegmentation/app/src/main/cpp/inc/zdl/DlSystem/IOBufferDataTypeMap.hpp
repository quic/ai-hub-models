//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include <cstddef>

#include "DlEnums.hpp"


#include "DlSystem/IOBufferDataTypeMap.h"

namespace DlSystem {

class IOBufferDataTypeMap : public Wrapper<IOBufferDataTypeMap, Snpe_IOBufferDataTypeMap_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_IOBufferDataTypeMap_Delete};

public:

  IOBufferDataTypeMap()
    : BaseType(Snpe_IOBufferDataTypeMap_Create())
  {  }

  void add(const char* name, IOBufferDataType_t bufferDataType){
    Snpe_IOBufferDataTypeMap_Add(handle(), name, static_cast<Snpe_IOBufferDataType_t>(bufferDataType));
  }

  void remove(const char* name){
    Snpe_IOBufferDataTypeMap_Remove(handle(), name);
  }

  IOBufferDataType_t getBufferDataType(const char* name){
    return static_cast<IOBufferDataType_t>(Snpe_IOBufferDataTypeMap_GetBufferDataType(handle(), name));
  }

  IOBufferDataType_t getBufferDataType(){
    return static_cast<IOBufferDataType_t>(Snpe_IOBufferDataTypeMap_GetBufferDataTypeOfFirst(handle()));
  }

  size_t size() const{
    return Snpe_IOBufferDataTypeMap_Size(handle());
  }

  bool find(const char* name) const{
    return Snpe_IOBufferDataTypeMap_Find(handle(), name);
  }

  void clear(){
    Snpe_IOBufferDataTypeMap_Clear(handle());
  }

  bool empty() const{
    return Snpe_IOBufferDataTypeMap_Empty(handle());
  }
};

} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, IOBufferDataTypeMap)
