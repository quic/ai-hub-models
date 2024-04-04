//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>

#include "Wrapper.hpp"

#include "DlSystem/TensorShape.h"

namespace DlSystem {


using Dimension = size_t;



class TensorShape : public Wrapper<TensorShape, Snpe_TensorShape_Handle_t> {
  friend BaseType;
  using BaseType::BaseType;

protected:
  static constexpr DeleteFunctionType DeleteFunction{Snpe_TensorShape_Delete};

private:
  using DimensionReference = WrapperDetail::MemberIndexedReference<TensorShape, Snpe_TensorShape_Handle_t, size_t, size_t, Snpe_TensorShape_At, Snpe_TensorShape_Set>;
  friend DimensionReference;

public:

  TensorShape()
    : BaseType(Snpe_TensorShape_Create())
  {  }

  TensorShape(const TensorShape& other)
    : BaseType(Snpe_TensorShape_CreateCopy(other.handle()))
  {  }

  TensorShape(TensorShape&& other) noexcept
    : BaseType(std::move(other))
  {  }

  TensorShape(std::initializer_list<Dimension> dims)
    : BaseType(Snpe_TensorShape_CreateDimsSize(dims.begin(), dims.size()))
  {  }

  TensorShape& operator=(const TensorShape& other) noexcept{
    if(this != &other){
      Snpe_TensorShape_Assign(other.handle(), handle());
    }
    return *this;
  }

  TensorShape& operator=(TensorShape&& other) noexcept{
    return moveAssign(std::move(other));
  }

  TensorShape(const size_t *dims, size_t size)
    : BaseType(Snpe_TensorShape_CreateDimsSize(dims, size))
  {  }

  TensorShape(const std::vector<size_t>& dims)
    : TensorShape(dims.data(), dims.size())
  {  }


  void concatenate(const size_t *dims, size_t size){
    Snpe_TensorShape_Concatenate(handle(), dims, size);
  }

  void concatenate(const size_t &dim){
    return concatenate(&dim, 1);
  }

  size_t operator[](size_t idx) const{
    return Snpe_TensorShape_At(handle(), idx);
  }

  DimensionReference operator[](size_t idx){
    return {*this, idx};
  }

  size_t rank() const{
    return Snpe_TensorShape_Rank(handle());
  }

  const size_t* getDimensions() const{
    return Snpe_TensorShape_GetDimensions(handle());
  }


};

} // ns DlSystem

ALIAS_IN_ZDL_NAMESPACE(DlSystem, Dimension)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, TensorShape)
