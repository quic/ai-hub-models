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
#include "TensorShape.hpp"

#include "DlSystem/IBufferAttributes.h"
#include "IUserBuffer.hpp"

namespace DlSystem {


class IBufferAttributes : public Wrapper<IBufferAttributes, Snpe_IBufferAttributes_Handle_t, true> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_IBufferAttributes_Delete};
public:

  size_t getElementSize() const noexcept{
    return Snpe_IBufferAttributes_GetElementSize(handle());
  }

  UserBufferEncoding::ElementType_t getEncodingType() const noexcept{
    return static_cast<UserBufferEncoding::ElementType_t>(Snpe_IBufferAttributes_GetEncodingType(handle()));
  }

  TensorShape getDims() const{
    return moveHandle(Snpe_IBufferAttributes_GetDims(handle()));
  }

  TensorShape getAlignments() const{
    return moveHandle(Snpe_IBufferAttributes_GetAlignments(handle()));
  }

  UserBufferEncoding* getEncoding() const{
    auto h = Snpe_IBufferAttributes_GetEncoding_Ref(handle());
    switch(Snpe_UserBufferEncoding_GetElementType(h)){
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        return makeReference<UserBufferEncodingFloat>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNSIGNED8BIT:
        return makeReference<UserBufferEncodingUnsigned8Bit>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
        return makeReference<UserBufferEncodingUintN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
        return makeReference<UserBufferEncodingIntN>(h);


      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16:
        return makeReference<UserBufferEncodingFloatN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        return makeReference<UserBufferEncodingTf8>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
        return makeReference<UserBufferEncodingTfN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8:
        return makeReference<UserBufferEncodingBool>(h);

      default:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN:
        return makeReference<UserBufferEncoding>(h);
    }
  }

};

} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, IBufferAttributes)
