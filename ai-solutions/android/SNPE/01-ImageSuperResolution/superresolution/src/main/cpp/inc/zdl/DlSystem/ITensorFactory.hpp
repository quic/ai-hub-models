//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"
#include "ITensor.hpp"

#include <istream>


#include "SNPE/SNPEUtil.h"

namespace DlSystem{
// NOTE: These factories use a different handle type because they are singletons
// Never copy this pattern unless you're also implementing a singleton
class ITensorFactory : public Wrapper<ITensorFactory, ITensorFactory*, true>{
  friend BaseType;

  using BaseType::BaseType;
  static constexpr DeleteFunctionType DeleteFunction{NoOpDeleter};

public:
  ITensorFactory()
    : BaseType(nullptr)
  {  }


  std::unique_ptr<ITensor> createTensor(const TensorShape &shape) noexcept{
    return makeUnique<ITensor>(Snpe_Util_CreateITensor(getHandle(shape)));
  }

  // Create from std::istream is no longer supported
  std::unique_ptr<ITensor> createTensor(std::istream &input) noexcept = delete;

  std::unique_ptr<ITensor> createTensor(const TensorShape &shape,
                                        const unsigned char *data,
                                        size_t dataSize) noexcept{
    auto handle = Snpe_Util_CreateITensorDataSize(getHandle(shape), data, dataSize);
    return makeUnique<ITensor>(handle);
  }

};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, ITensorFactory)
