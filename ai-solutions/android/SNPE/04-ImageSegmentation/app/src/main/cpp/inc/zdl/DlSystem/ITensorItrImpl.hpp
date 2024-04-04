//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once
#include "Wrapper.hpp"

namespace DlSystem {

class ITensorItrImpl {
public:
  ITensorItrImpl() = default;
  virtual ~ITensorItrImpl() = default;

  virtual float getValue() const = 0;
  virtual float& getReference() = 0;
  virtual float& getReferenceAt(size_t idx) = 0;
  virtual float* dataPointer() const = 0;
  virtual void increment(int incVal = 1) = 0;
  virtual void decrement(int decVal = 1) = 0;
  virtual size_t getPosition() = 0;
  virtual std::unique_ptr<DlSystem::ITensorItrImpl> clone() = 0;

private:
  ITensorItrImpl& operator=(const ITensorItrImpl& other) = delete;
  ITensorItrImpl(const ITensorItrImpl& other) = delete;
};

} // ns DlSystem
