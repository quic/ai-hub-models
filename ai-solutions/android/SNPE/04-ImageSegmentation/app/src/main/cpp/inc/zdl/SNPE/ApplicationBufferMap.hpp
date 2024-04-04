//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

#include "Wrapper.hpp"
#include "DlSystem/StringList.hpp"

#include "SNPE/ApplicationBufferMap.h"

namespace PSNPE {

class ApplicationBufferMap : public Wrapper<ApplicationBufferMap, Snpe_ApplicationBufferMap_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_ApplicationBufferMap_Delete};
public:
  ApplicationBufferMap()
  : BaseType(Snpe_ApplicationBufferMap_Create()){}

  explicit ApplicationBufferMap(const std::unordered_map<std::string, std::vector<uint8_t>> &buffer)
  : ApplicationBufferMap(){
    for(const auto &kv: buffer){
      add(kv.first.c_str(), kv.second);
    }
  }

  void add(const char *name, const std::vector<uint8_t> &buff){
    Snpe_ApplicationBufferMap_Add(handle(), name, buff.data(), buff.size());
  }

  void add(const char *name, const std::vector<float> &buff){
    Snpe_ApplicationBufferMap_Add(handle(), name, reinterpret_cast<const uint8_t *>(buff.data()), buff.size()*sizeof(float));
  }

  void remove(const char *name) noexcept{
    Snpe_ApplicationBufferMap_Remove(handle(), name);
  }

  size_t size() const noexcept{
    return Snpe_ApplicationBufferMap_Size(handle());
  }

  void clear() noexcept{
    Snpe_ApplicationBufferMap_Clear(handle());
  }

  std::vector<uint8_t> getUserBuffer(const char *name) const{
    size_t size{};
    const uint8_t *data{};
    Snpe_ApplicationBufferMap_GetUserBuffer(handle(), name, &size, &data);

    return std::vector<uint8_t>(data, data + size);
  }

  std::vector<uint8_t> operator[](const char *name) const{
    return getUserBuffer(name);
  }

  DlSystem::StringList getUserBufferNames() const{
    return moveHandle(Snpe_ApplicationBufferMap_GetUserBufferNames(handle()));
  }

  std::unordered_map<std::string, std::vector<uint8_t>> getUserBuffer() const{
    std::unordered_map<std::string, std::vector<uint8_t>> toret;
    for(auto name: getUserBufferNames()){
      toret.emplace(name, getUserBuffer(name));
    }

    return toret;
  }

};

} // ns PSNPE


ALIAS_IN_ZDL_NAMESPACE(PSNPE, ApplicationBufferMap)
