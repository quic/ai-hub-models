//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <type_traits>
#include <algorithm>
#include <iterator>

#include "Wrapper.hpp"
#include "ITensorItrImpl.hpp"

namespace DlSystem{

template<bool IS_CONST=true>
class ITensorItr{
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using pointer = typename std::conditional<IS_CONST, const float*, float*>::type;
  using value_type = float;
  using difference_type = std::ptrdiff_t;
  using reference = typename std::conditional<IS_CONST, const float&, float&>::type;


  ITensorItr() = delete;
  virtual ~ITensorItr() = default;

  explicit ITensorItr(pointer data) noexcept
    : m_Impl{nullptr},
      m_IsTrivial{true},
      m_Data{data},
      m_DataStart{data}
  {  }

  ITensorItr(std::unique_ptr<ITensorItrImpl> impl,
             bool isTrivial = false,
             float* data = nullptr)
    : m_Impl(impl->clone()),
      m_IsTrivial(isTrivial),
      m_Data(data),
      m_DataStart(data)
  {  }

  ITensorItr(const ITensorItr& itr)
    : m_Impl(itr.m_Impl ? itr.m_Impl->clone() : nullptr),
      m_IsTrivial(itr.m_IsTrivial),
      m_Data(itr.m_Data),
      m_DataStart(itr.m_DataStart)
  {  }

  ITensorItr(ITensorItr&& itr) noexcept
    : m_Impl(std::move(itr.m_Impl)),
      m_IsTrivial(itr.m_IsTrivial),
      m_Data(itr.m_Data),
      m_DataStart(itr.m_DataStart)
  {  }

  ITensorItr& operator=(const ITensorItr& other){
    if (this == &other) return *this;

    m_Impl = other.m_Impl ? other.m_Impl->clone() : nullptr;
    m_IsTrivial = other.m_IsTrivial;
    m_Data = other.m_Data;
    m_DataStart = other.m_DataStart;
    return *this;
  }
  ITensorItr& operator=(ITensorItr&& other) noexcept{
    if(this != &other){
      m_Impl = std::move(other.m_Impl);
      m_IsTrivial = other.m_IsTrivial;
      m_Data = other.m_Data;
      m_DataStart = other.m_DataStart;
    }
    return *this;
  }

  inline ITensorItr& operator++(){
    if (m_IsTrivial){
      m_Data++;
    } else {
      m_Impl->increment();
    }
    return *this;
  }
  inline ITensorItr operator++(int){
    ITensorItr tmp(*this);
    operator++();
    return tmp;
  }
  inline ITensorItr& operator--(){
    if (m_IsTrivial){
      m_Data--;
    } else {
      m_Impl->decrement();
    }
    return *this;
  }
  inline ITensorItr operator--(int){
    ITensorItr tmp(*this);
    operator--();
    return tmp;
  }
  inline ITensorItr& operator+=(int rhs){
    if (m_IsTrivial){
      m_Data += rhs;
    } else {
      m_Impl->increment(rhs);
    }
    return *this;
  }
  inline friend ITensorItr operator+(ITensorItr lhs, int rhs){
    lhs += rhs;
    return lhs;
  }
  inline ITensorItr& operator-=(int rhs){
   if (m_IsTrivial){
     m_Data -= rhs;
   } else {
     m_Impl->decrement(rhs);
   }
   return *this;
  }
  inline friend ITensorItr operator-(ITensorItr lhs, int rhs){
    lhs -= rhs;
    return lhs;
  }

  inline size_t operator-(const ITensorItr& rhs){
    if (m_IsTrivial) return (m_Data - m_DataStart) - (rhs.m_Data - rhs.m_DataStart);
    return m_Impl->getPosition() - rhs.m_Impl->getPosition();
  }

  inline friend bool operator<(const ITensorItr& lhs, const ITensorItr& rhs){
    if (lhs.m_IsTrivial) return lhs.m_Data < rhs.m_Data;
    return lhs.m_Impl->dataPointer() < rhs.m_Impl->dataPointer();
  }
  inline friend bool operator>(const ITensorItr& lhs, const ITensorItr& rhs){
    return rhs < lhs;
  }
  inline friend bool operator<=(const ITensorItr& lhs, const ITensorItr& rhs){
    return !(lhs > rhs);
  }
  inline friend bool operator>=(const ITensorItr& lhs, const ITensorItr& rhs){
    return !(lhs < rhs);
  }

  inline bool operator==(const ITensorItr& rhs) const{
    if (m_IsTrivial) return m_Data == rhs.m_Data;
    return m_Impl->dataPointer() == rhs.m_Impl->dataPointer();
  }
  inline bool operator!=(const ITensorItr& rhs) const{
    return !operator==(rhs);
  }

  inline reference operator[](size_t idx){
    if (m_IsTrivial) return *(m_DataStart + idx);
    return m_Impl->getReferenceAt(idx);
  }
  inline reference operator*(){
    if (m_IsTrivial) return *m_Data;
    return m_Impl->getReference();
  }
  inline reference operator->(){
    return *(*this);
  }
  inline float* dataPointer() const{
    if (m_IsTrivial) return m_Data;
    return m_Impl->dataPointer();
  }


protected:
  std::unique_ptr<::DlSystem::ITensorItrImpl> m_Impl;
  bool m_IsTrivial = false;
  pointer m_Data = nullptr;
  pointer m_DataStart = nullptr;
};


inline void fill(ITensorItr<false> first, ITensorItr<false> end, float val){
  std::fill(first, end, val);
}
template<class InItr, class OutItr>
OutItr copy(InItr first, InItr last, OutItr result){
  return std::copy(first, last, result);
}

} // ns DlSystem


// ALIAS_IN_ZDL_NAMESPACE
namespace zdl{ namespace DlSystem{
  template<bool IS_CONST>
  using ITensorItr = ::DlSystem::ITensorItr<IS_CONST>;
}}
