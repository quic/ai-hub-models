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

#include "DlSystem/IUserBuffer.h"


namespace DlSystem {


class UserBufferEncoding: public Wrapper<UserBufferEncoding, Snpe_UserBufferEncoding_Handle_t> {
  friend BaseType;
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_UserBufferEncoding_Delete};
protected:
  UserBufferEncoding(HandleType handle)
    : BaseType(handle)
  {  }
public:

  virtual ~UserBufferEncoding() = default;

  UserBufferEncoding(UserBufferEncoding&& other) noexcept
    : BaseType(std::move(other))
  {  }

  enum class ElementType_t
  {
    /// Unknown element type.
    UNKNOWN         = 0,

    /// Each element is presented by 32-bit float.
    FLOAT           = 1,

    /// Each element is presented by an unsigned int.
    UNSIGNED8BIT    = 2,

    /// Each element is presented by 16-bit float.
    FLOAT16         = 3,

    /// Each element is presented by an 8-bit quantized value.
    TF8             = 10,

    /// Each element is presented by an 16-bit quantized value.
    TF16            = 11,

    /// Each element is presented by Int32
    INT32           = 12,

    /// Each element is presented by UInt32
    UINT32          = 13,

    /// Each element is presented by Int8
    INT8            = 14,

    /// Each element is presented by UInt8
    UINT8           = 15,

    /// Each element is presented by Int16
    INT16           = 16,

    /// Each element is presented by UInt16
    UINT16          = 17,

    // Each element is presented by Bool8
    BOOL8           = 18,

    // Each element is presented by Int64
    INT64           = 19,

    // Each element is presented by UInt64
    UINT64           = 20
  };

  ElementType_t getElementType() const noexcept{
    return static_cast<ElementType_t>(Snpe_UserBufferEncoding_GetElementType(handle()));
  }

  size_t getElementSize() const noexcept{
    return Snpe_UserBufferEncoding_GetElementSize(handle());
  }
};


class UserBufferSource: public Wrapper<UserBufferSource, Snpe_UserBufferSource_Handle_t> {
  friend BaseType;
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_UserBufferSource_Delete};

public:
  enum class SourceType_t
  {
    /// Unknown buffer source type.
    UNKNOWN = 0,

    /// The network inputs are from CPU buffer.
    CPU = 1,

    /// The network inputs are from OpenGL buffer.
    GLBUFFER = 2
  };
protected:
  UserBufferSource(HandleType handle)
    : BaseType(handle)
  {  }
public:
  SourceType_t getSourceType() const noexcept{
    return static_cast<SourceType_t>(Snpe_UserBufferSource_GetSourceType(handle()));
  }

};

class UserBufferSourceGLBuffer : public UserBufferSource{
public:
  UserBufferSourceGLBuffer()
    : UserBufferSource(Snpe_UserBufferSourceGLBuffer_Create())
  {  }
};

class UserBufferEncodingUnsigned8Bit : public UserBufferEncoding{
public:
  using UserBufferEncoding::UserBufferEncoding;
  UserBufferEncodingUnsigned8Bit()
    : UserBufferEncoding(Snpe_UserBufferEncodingUnsigned8Bit_Create())
  {  }
};

class UserBufferEncodingFloatN : public UserBufferEncoding{
public:
  using UserBufferEncoding::UserBufferEncoding;

  UserBufferEncodingFloatN(uint8_t bWidth=32)
    : UserBufferEncoding(Snpe_UserBufferEncodingFloatN_Create(bWidth))
  {  }

  UserBufferEncodingFloatN(const UserBufferEncodingFloatN& other)
    : UserBufferEncoding(Snpe_UserBufferEncodingFloatN_CreateCopy(other.handle()))
  {  }

  static ElementType_t getTypeFromWidth(uint8_t width){
    return static_cast<ElementType_t>(Snpe_UserBufferEncodingFloatN_GetTypeFromWidth(width));
  }
};

class UserBufferEncodingFloat : public UserBufferEncoding{
public:
  using UserBufferEncoding::UserBufferEncoding;
  UserBufferEncodingFloat()
    : UserBufferEncoding(Snpe_UserBufferEncodingFloat_Create())
  {  }
  UserBufferEncodingFloat(const UserBufferEncodingFloat& other)
    :  UserBufferEncoding(Snpe_UserBufferEncodingFloat_CreateCopy(other.handle()))
  {  }

  UserBufferEncodingFloat(UserBufferEncodingFloat&& other) noexcept
    : UserBufferEncoding(std::move(other))
  {  }
};


class UserBufferEncodingTfN : public UserBufferEncoding{
public:

  using UserBufferEncoding::UserBufferEncoding;
  template<typename T, typename U,
    typename std::enable_if<std::is_integral<T>::value && std::is_floating_point<U>::value, int>::type = 0>
  UserBufferEncodingTfN(T stepFor0, U stepSize, uint8_t bWidth=8)
    : UserBufferEncoding(Snpe_UserBufferEncodingTfN_Create(stepFor0, stepSize, bWidth))
  {  }

  UserBufferEncodingTfN(const UserBufferEncoding& ubEncoding)
    : UserBufferEncoding(Snpe_UserBufferEncodingTfN_CreateCopy(getHandle(ubEncoding)))
  {  }
  UserBufferEncodingTfN(const UserBufferEncodingTfN& ubEncoding)
    : UserBufferEncoding(Snpe_UserBufferEncodingTfN_CreateCopy(getHandle(ubEncoding)))
  {  }

  void setStepExactly0(uint64_t stepExactly0){
    Snpe_UserBufferEncodingTfN_SetStepExactly0(handle(), stepExactly0);
  }

  void setQuantizedStepSize(const float quantizedStepSize){
    Snpe_UserBufferEncodingTfN_SetQuantizedStepSize(handle(), quantizedStepSize);
  }

  uint64_t getStepExactly0() const{
    return Snpe_UserBufferEncodingTfN_GetStepExactly0(handle());
  }

  float getMin() const{
    return Snpe_UserBufferEncodingTfN_GetMin(handle());
  }
  float getMax() const{
    return Snpe_UserBufferEncodingTfN_GetMax(handle());
  }

  float getQuantizedStepSize() const{
    return Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(handle());
  }

  static ElementType_t getTypeFromWidth(uint8_t width){
    return static_cast<ElementType_t>(Snpe_UserBufferEncodingTfN_GetTypeFromWidth(width));
  }
};

class UserBufferEncodingIntN : public UserBufferEncoding{
public:

  UserBufferEncodingIntN(uint8_t bWidth=32)
    : UserBufferEncoding(Snpe_UserBufferEncodingIntN_Create(bWidth))
  {  }

  UserBufferEncodingIntN(const UserBufferEncoding& ubEncoding)
    : UserBufferEncoding(Snpe_UserBufferEncodingIntN_CreateCopy(getHandle(ubEncoding)))
  {  }

  static ElementType_t getTypeFromWidth(uint8_t width){
    return static_cast<ElementType_t>(Snpe_UserBufferEncodingIntN_GetTypeFromWidth(width));
  }
};



class UserBufferEncodingUintN : public UserBufferEncoding{
public:

  UserBufferEncodingUintN(uint8_t bWidth=32)
    : UserBufferEncoding(Snpe_UserBufferEncodingUintN_Create(bWidth))
  {  }

  UserBufferEncodingUintN(const UserBufferEncoding& ubEncoding)
    : UserBufferEncoding(Snpe_UserBufferEncodingUintN_CreateCopy(getHandle(ubEncoding)))
  {  }

  static ElementType_t getTypeFromWidth(uint8_t width){
    return static_cast<ElementType_t>(Snpe_UserBufferEncodingUintN_GetTypeFromWidth(width));
  }
};


class UserBufferEncodingTf8 : public UserBufferEncodingTfN{
public:
  using UserBufferEncodingTfN::UserBufferEncodingTfN;
  UserBufferEncodingTf8() = delete;

  template<typename T, typename U,
    typename std::enable_if<std::is_integral<T>::value && std::is_floating_point<U>::value, int>::type = 0>
  UserBufferEncodingTf8(T stepFor0, U stepSize)
    : UserBufferEncodingTfN(stepFor0, stepSize, 8)
  {  }

  UserBufferEncodingTf8(const UserBufferEncoding& ubEncoding)
    : UserBufferEncodingTfN(ubEncoding)
  {  }

};

class UserBufferEncodingBool : public UserBufferEncoding{
public:
    UserBufferEncodingBool(uint8_t bWidth=8)
      : UserBufferEncoding(Snpe_UserBufferEncodingBool_Create(bWidth))
    {  }

    UserBufferEncodingBool(const UserBufferEncoding& ubEncoding)
      : UserBufferEncoding(Snpe_UserBufferEncodingBool_CreateCopy(getHandle(ubEncoding)))
    {  }
};

class IUserBuffer: public Wrapper<IUserBuffer, Snpe_IUserBuffer_Handle_t, true> {
  friend BaseType;
  using BaseType::BaseType;
  static constexpr DeleteFunctionType DeleteFunction{Snpe_IUserBuffer_Delete};

public:
  const TensorShape& getStrides() const{
    return *makeReference<TensorShape>(Snpe_IUserBuffer_GetStrides_Ref(handle()));
  }

  size_t getSize() const{
    return Snpe_IUserBuffer_GetSize(handle());
  }

  size_t getOutputSize() const{
    return Snpe_IUserBuffer_GetOutputSize(handle());
  }

  bool setBufferAddress(void* buffer) noexcept{
    return Snpe_IUserBuffer_SetBufferAddress(handle(), buffer);
  }

  const UserBufferEncoding& getEncoding() const noexcept{
    auto h = Snpe_IUserBuffer_GetEncoding_Ref(handle());
    switch(Snpe_UserBufferEncoding_GetElementType(h)){
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        return *makeReference<UserBufferEncodingFloat>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNSIGNED8BIT:
        return *makeReference<UserBufferEncodingUnsigned8Bit>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
        return *makeReference<UserBufferEncodingUintN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
        return *makeReference<UserBufferEncodingIntN>(h);


      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16:
        return *makeReference<UserBufferEncodingFloatN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        return *makeReference<UserBufferEncodingTf8>(h);
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
        return *makeReference<UserBufferEncodingTfN>(h);
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8:
        return *makeReference<UserBufferEncodingBool>(h);

      default:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN:
        return *makeReference<UserBufferEncoding>(h);
    }
  }
  UserBufferEncoding& getEncoding() noexcept{
    auto h = Snpe_IUserBuffer_GetEncoding_Ref(handle());
    switch(Snpe_UserBufferEncoding_GetElementType(h)){
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        return *makeReference<UserBufferEncodingFloat>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNSIGNED8BIT:
        return *makeReference<UserBufferEncodingUnsigned8Bit>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
        return *makeReference<UserBufferEncodingUintN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
        return *makeReference<UserBufferEncodingIntN>(h);


      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16:
        return *makeReference<UserBufferEncodingFloatN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        return *makeReference<UserBufferEncodingTf8>(h);
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
        return *makeReference<UserBufferEncodingTfN>(h);

      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_BOOL8:
        return *makeReference<UserBufferEncodingBool>(h);

      default:
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN:
        return *makeReference<UserBufferEncoding>(h);
    }
  }

};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncoding)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferSource)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferSourceGLBuffer)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingUnsigned8Bit)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingFloatN)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingFloat)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingTfN)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingIntN)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingUintN)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, UserBufferEncodingTf8)

ALIAS_IN_ZDL_NAMESPACE(DlSystem, IUserBuffer)
