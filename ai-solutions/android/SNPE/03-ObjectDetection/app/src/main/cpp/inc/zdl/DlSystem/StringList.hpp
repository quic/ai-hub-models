//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "DlSystem/DlError.hpp"

#include "DlSystem/StringList.h"


namespace DlSystem {

class StringList : public Wrapper<StringList, Snpe_StringList_Handle_t>{
  friend BaseType;
  using BaseType::BaseType;
  static constexpr DeleteFunctionType DeleteFunction = Snpe_StringList_Delete;

public:
  StringList()
    : BaseType(Snpe_StringList_Create())
  {  }
  explicit StringList(size_t length)
    : BaseType(Snpe_StringList_CreateSize(length))
  {  }
  StringList(const StringList& other)
    : BaseType(Snpe_StringList_CreateCopy(other.handle()))
  {  }
  StringList(StringList&& other) noexcept
    : BaseType(std::move(other))
  {  }


  StringList& operator=(const StringList& other){
    if(this != &other){
      Snpe_StringList_Assign(other.handle(), handle());
    }
    return *this;
  }
  StringList& operator=(StringList&& other) noexcept{
    return moveAssign(std::move(other));
  }


  DlSystem::ErrorCode append(const char* str){
    return static_cast<DlSystem::ErrorCode>(Snpe_StringList_Append(handle(), str));
  }

  const char* at(size_t idx) const noexcept{
    return Snpe_StringList_At(handle(), idx);
  }

  const char** begin() const noexcept{
    return Snpe_StringList_Begin(handle());
  }
  const char** end() const noexcept{
    return Snpe_StringList_End(handle());
  }

  size_t size() const noexcept{
    return Snpe_StringList_Size(handle());
  }

};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, StringList)
