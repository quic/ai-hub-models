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
#include "DlSystem/StringList.hpp"

#include "DlSystem/UserMemoryMap.h"

namespace DlSystem {

class UserMemoryMap : public Wrapper<UserMemoryMap, Snpe_UserMemoryMap_Handle_t> {
  friend BaseType;
// Use this to get free move Ctor and move assignment operator, provided this class does not specify
// as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_UserMemoryMap_Delete};
public:
  UserMemoryMap()
    : BaseType(Snpe_UserMemoryMap_Create())
  {  }
  UserMemoryMap(const UserMemoryMap& other)
    : BaseType(Snpe_UserMemoryMap_Copy(other.handle()))
  {  }
  UserMemoryMap(UserMemoryMap&& other) noexcept
    : BaseType(std::move(other))
  {  }

  UserMemoryMap& operator=(const UserMemoryMap& other){
    if(this != &other){
      Snpe_UserMemoryMap_Assign(handle(), other.handle());
    }
    return *this;
  }

  DlSystem::ErrorCode add(const char* name, void* address) noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_UserMemoryMap_Add(handle(), name, address));
  }

  DlSystem::ErrorCode remove(const char* name){
    return static_cast<DlSystem::ErrorCode>(Snpe_UserMemoryMap_Remove(handle(), name));
  }

  size_t size() const noexcept{
    return Snpe_UserMemoryMap_Size(handle());
  }

  DlSystem::ErrorCode clear() noexcept{
    return static_cast<DlSystem::ErrorCode>(Snpe_UserMemoryMap_Clear(handle()));
  }

  StringList getUserBufferNames() const{
    return moveHandle(Snpe_UserMemoryMap_GetUserBufferNames(handle()));
  }

  size_t getUserMemoryAddressCount(const char* name) const noexcept{
    return Snpe_UserMemoryMap_GetUserMemoryAddressCount(handle(), name);
  }

  void* getUserMemoryAddressAtIndex(const char* name, uint32_t index) const noexcept{
    return Snpe_UserMemoryMap_GetUserMemoryAddressAtIndex(handle(), name, index);
  }

};


} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserMemoryMap)
