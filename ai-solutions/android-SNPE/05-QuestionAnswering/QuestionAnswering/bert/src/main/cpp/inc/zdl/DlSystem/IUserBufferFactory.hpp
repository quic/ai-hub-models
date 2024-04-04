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
#include "IUserBuffer.hpp"
#include "TensorShape.hpp"


#include "SNPE/SNPEUtil.h"

namespace DlSystem{


// NOTE: These factories use a different handle type because they are singletons
// Never copy this pattern unless you're also implementing a singleton
class IUserBufferFactory : public Wrapper<IUserBufferFactory, IUserBufferFactory*, true>{
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{NoOpDeleter};

public:
  IUserBufferFactory()
    : BaseType(nullptr)
  {  }

  std::unique_ptr<IUserBuffer> createUserBuffer(void *buffer,
                                                size_t bufSize,
                                                const TensorShape &strides,
                                                UserBufferEncoding* userBufferEncoding) noexcept{
    if(!userBufferEncoding) return {};
    auto handle = Snpe_Util_CreateUserBuffer(buffer,
                                             bufSize,
                                             getHandle(strides),
                                             getHandle(userBufferEncoding));
    return makeUnique<IUserBuffer>(handle);
  }

  std::unique_ptr<IUserBuffer> createUserBuffer(void *buffer,
                                                size_t bufSize,
                                                const TensorShape &strides,
                                                UserBufferEncoding* userBufferEncoding,
                                                UserBufferSource* userBufferSource) noexcept{
    if(!userBufferEncoding || !userBufferSource) return {};
    auto handle = Snpe_Util_CreateUserBufferFromSource(buffer,
                                                       bufSize,
                                                       getHandle(strides),
                                                       getHandle(*userBufferEncoding),
                                                       getHandle(*userBufferSource));
    return makeUnique<IUserBuffer>(handle);
  }

};


} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, IUserBufferFactory)
