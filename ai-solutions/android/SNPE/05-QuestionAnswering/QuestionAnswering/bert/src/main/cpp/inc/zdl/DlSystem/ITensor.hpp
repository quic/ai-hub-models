//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "TensorShape.hpp"
#include "ITensorItr.hpp"

#include "DlSystem/ITensor.h"


namespace DlSystem {


class ITensor : public Wrapper<ITensor, Snpe_ITensor_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_ITensor_Delete};

  template<typename T>
  T* getData(){
    return static_cast<T*>(Snpe_ITensor_GetData(handle()));
  }

  template<typename T>
  const T* getData() const{
    return static_cast<const T*>(Snpe_ITensor_GetData(handle()));
  }

public:
  using iterator = DlSystem::ITensorItr<false>;
  using const_iterator = DlSystem::ITensorItr<true>;


  iterator begin(){
    return iterator(getData<float>());
  }

  const_iterator begin() const{
    return const_iterator(getData<float>());
  }

  const_iterator cbegin() const{
    return begin();
  }

  iterator end(){
    return begin() + getSize();
  }

  const_iterator end() const{
    return cbegin() + getSize();
  }

  const_iterator cend() const{
    return end();
  }

  TensorShape getShape() const{
    return moveHandle(Snpe_ITensor_GetShape(handle()));
  }

  size_t getSize() const{
    return Snpe_ITensor_GetSize(handle());
  }

  // Serialize to std::ostream is no longer supported
  void serialize(std::ostream &output) const = delete;

  bool isQuantized() const{
    return Snpe_ITensor_IsQuantized(handle());
  }

  float GetDelta() const{
    return Snpe_ITensor_GetDelta(handle());
  }

  float GetOffset() const{
    return Snpe_ITensor_GetOffset(handle());
  }
};


} //ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, ITensor)
