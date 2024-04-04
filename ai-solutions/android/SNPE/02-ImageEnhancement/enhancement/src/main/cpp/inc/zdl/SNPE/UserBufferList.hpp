//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "DlSystem/UserBufferMap.hpp"

#include "SNPE/UserBufferList.h"


namespace PSNPE {

class UserBufferList : public Wrapper<UserBufferList, Snpe_UserBufferList_Handle_t, true> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_UserBufferList_Delete};

public:
  UserBufferList()
    : BaseType(Snpe_UserBufferList_Create())
  {  }
  explicit UserBufferList(size_t size)
    : BaseType(Snpe_UserBufferList_CreateSize(size))
  {  }

  UserBufferList(const UserBufferList& other)
    : BaseType(Snpe_UserBufferList_CreateCopy(other.handle()))
  {  }
  UserBufferList(UserBufferList&& other) noexcept
    : BaseType(std::move(other))
  {  }

  UserBufferList& operator=(const UserBufferList& other){
    if(this != &other){
      Snpe_UserBufferList_Assign(other.handle(), handle());
    }
    return *this;
  }
  UserBufferList& operator=(UserBufferList&& other){
    return moveAssign(std::move(other));
  }


  void push_back(const DlSystem::UserBufferMap&  userBufferMap){
    Snpe_UserBufferList_PushBack(handle(), getHandle(userBufferMap));
  }

  DlSystem::UserBufferMap& operator[](size_t idx){
    return *makeReference<DlSystem::UserBufferMap>(Snpe_UserBufferList_At_Ref(handle(), idx));
  }

  size_t size() const noexcept{
    return Snpe_UserBufferList_Size(handle());
  }

  size_t capacity() const noexcept{
    return Snpe_UserBufferList_Capacity(handle());
  }

  void clear() noexcept{
    Snpe_UserBufferList_Clear(handle());
  }
};


} // ns PSNPE

ALIAS_IN_ZDL_NAMESPACE(PSNPE, UserBufferList)
