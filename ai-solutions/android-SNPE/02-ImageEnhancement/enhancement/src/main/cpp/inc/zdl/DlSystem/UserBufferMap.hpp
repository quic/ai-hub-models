//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <cstddef>

#include "Wrapper.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/IUserBuffer.hpp"

#include "DlSystem/UserBufferMap.h"

namespace DlSystem {

class UserBufferMap : public Wrapper<UserBufferMap, Snpe_UserBufferMap_Handle_t, true> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_UserBufferMap_Delete};

public:
  UserBufferMap()
    : BaseType(Snpe_UserBufferMap_Create())
  {  }

  UserBufferMap(const UserBufferMap& other)
    : BaseType(Snpe_UserBufferMap_CreateCopy(other.handle()))
  {  }
  UserBufferMap(UserBufferMap&& other) noexcept
    : BaseType(std::move(other))
  {  }

  UserBufferMap& operator=(const UserBufferMap& other){
    if(this != &other){
      Snpe_UserBufferMap_Assign(other.handle(), handle());
    }
    return *this;
  }
  UserBufferMap& operator=(UserBufferMap&& other) noexcept{
    return moveAssign(std::move(other));
  }

  DlSystem::ErrorCode add(const char* name, IUserBuffer* buffer){
    if(!buffer) return ErrorCode::SNPE_CAPI_BAD_ARGUMENT;
    return static_cast<DlSystem::ErrorCode>(Snpe_UserBufferMap_Add(handle(), name, getHandle(*buffer)));
  }

  DlSystem::ErrorCode remove(const char* name) noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_UserBufferMap_Remove(handle(), name));
  }

  size_t size() const noexcept{
    return Snpe_UserBufferMap_Size(handle());
  }

  DlSystem::ErrorCode clear() noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_UserBufferMap_Clear(handle()));
  }

  IUserBuffer* getUserBuffer(const char* name) const noexcept{
    return makeReference<IUserBuffer>(Snpe_UserBufferMap_GetUserBuffer_Ref(handle(), name));
  }

  StringList getUserBufferNames() const{
    return moveHandle(Snpe_UserBufferMap_GetUserBufferNames(handle()));
  }

};

} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferMap)
