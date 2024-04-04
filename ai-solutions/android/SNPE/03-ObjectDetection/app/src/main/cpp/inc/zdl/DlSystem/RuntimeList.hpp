//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "StringList.hpp"
#include "DlEnums.hpp"
#include "DlSystem/RuntimeList.h"






namespace DlSystem {

class RuntimeList : public Wrapper<RuntimeList, Snpe_RuntimeList_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_RuntimeList_Delete};

  static Runtime_t GetRuntime(HandleType handle, size_t idx){
    return static_cast<Runtime_t>(Snpe_RuntimeList_GetRuntime(handle, int(idx)));
  }
  static Snpe_ErrorCode_t SetRuntime(HandleType handle, size_t idx, Runtime_t runtime){
    return Snpe_RuntimeList_SetRuntime(handle, idx, static_cast<Snpe_Runtime_t>(runtime));
  }

private:
  using RuntimeReference = WrapperDetail::MemberIndexedReference<RuntimeList, Snpe_RuntimeList_Handle_t, Runtime_t, size_t, GetRuntime, SetRuntime>;
  friend RuntimeReference;

public:

  RuntimeList()
    : BaseType(Snpe_RuntimeList_Create())
  {  }
  RuntimeList(const RuntimeList& other)
    : BaseType(Snpe_RuntimeList_CreateCopy(other.handle()))
  {  }
  RuntimeList(RuntimeList&& other) noexcept
    : BaseType(std::move(other))
  {  }

  RuntimeList(const Runtime_t& runtime)
    : BaseType(Snpe_RuntimeList_Create())
  {
    Snpe_RuntimeList_Add(handle(), static_cast<Snpe_Runtime_t>(runtime));
  }

  RuntimeList& operator=(const RuntimeList& other){
    if(this != &other){
      Snpe_RuntimeList_Assign(other.handle(), handle());
    }
    return *this;
  }

  RuntimeList& operator=(RuntimeList&& other) noexcept{
    return moveAssign(std::move(other));
  }

  Runtime_t operator[](size_t idx) const{
    return GetRuntime(handle(), idx);
  }

  RuntimeReference operator[](size_t idx) noexcept{
    return {*this, idx};
  }

  bool add(const Runtime_t& runtime){
    return SNPE_SUCCESS == Snpe_RuntimeList_Add(handle(), static_cast<Snpe_Runtime_t>(runtime));
  }

  void remove(Runtime_t runtime) noexcept{
    Snpe_RuntimeList_Remove(handle(), static_cast<Snpe_Runtime_t>(runtime));
  }

  size_t size() const noexcept{
    return Snpe_RuntimeList_Size(handle());
  }

  bool empty() const noexcept{
    return Snpe_RuntimeList_Empty(handle());
  }

  void clear() noexcept{
    Snpe_RuntimeList_Clear(handle());
  }

  StringList getRuntimeListNames() const{
    return moveHandle(Snpe_RuntimeList_GetRuntimeListNames(handle()));
  }

  static Runtime_t stringToRuntime(const char* runtimeStr){
    return static_cast<Runtime_t>(Snpe_RuntimeList_StringToRuntime(runtimeStr));
  }
  static const char* runtimeToString(Runtime_t runtime){
    return Snpe_RuntimeList_RuntimeToString(static_cast<Snpe_Runtime_t>(runtime));
  }

};


} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, RuntimeList)
