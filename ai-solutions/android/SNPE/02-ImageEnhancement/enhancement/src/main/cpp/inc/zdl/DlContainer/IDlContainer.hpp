//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <stdexcept>

#include "Wrapper.hpp"
#include "DlSystem/String.hpp"

#include "DlContainer/DlContainer.h"
#include "DlSystem/StringList.hpp"



namespace DlContainer {

struct DlcRecord
{
  std::string name;
  std::vector<uint8_t> data;

  DlcRecord()
    : name{},
      data{}
  {  }

  DlcRecord( DlcRecord&& other ) noexcept
    : name(std::move(other.name)),
     data(std::move(other.data))
  {  }
  DlcRecord(const std::string& new_name)
    : name(new_name),
      data()
  {
    if(name.empty()) {
      name.reserve(1);
    }
  }
  DlcRecord(const DlcRecord&) = delete;
};


class IDlContainer : public Wrapper<IDlContainer, Snpe_DlContainer_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_DlContainer_Delete};

  template<typename StringType>
  void getCatalog_(std::set<StringType>& catalog) const{
    DlSystem::StringList sl(moveHandle(Snpe_DlContainer_GetCatalog(handle())));
    for(auto s : sl){
      catalog.emplace(s);
    }
  }


  class DlcRecordInternal : public Wrapper<DlcRecordInternal, Snpe_DlcRecord_Handle_t> {
    friend BaseType;
    using BaseType::BaseType;

    static constexpr DeleteFunctionType DeleteFunction{Snpe_DlcRecord_Delete};
  public:
    DlcRecordInternal()
      : BaseType(Snpe_DlcRecord_Create())
    {  }
    explicit DlcRecordInternal(const std::string& name)
    : BaseType(Snpe_DlcRecord_CreateName(name.c_str()))
    {  }

    uint8_t* getData(){
      return Snpe_DlcRecord_Data(handle());
    }
    size_t size() const{
      return Snpe_DlcRecord_Size(handle());
    }
    const char* getName(){
      return Snpe_DlcRecord_Name(handle());
    }
  };


public:
  static std::unique_ptr<IDlContainer> open(const std::string& filename) noexcept{
    return makeUnique<IDlContainer>(Snpe_DlContainer_Open(filename.c_str()));
  }

  static std::unique_ptr<IDlContainer> open(const uint8_t* buffer, const size_t size) noexcept{
    return makeUnique<IDlContainer>(Snpe_DlContainer_OpenBuffer(buffer, size));

  }
  static std::unique_ptr<IDlContainer> open(const std::vector<uint8_t>& buffer) noexcept{
    return open(buffer.data(), buffer.size());
  }
  static std::unique_ptr<IDlContainer> open(const DlSystem::String &filename) noexcept{
    return open(static_cast<const std::string&>(filename));
  }


  void getCatalog(std::set<std::string>& catalog) const{
    return getCatalog_(catalog);
  }
  void getCatalog(std::set<DlSystem::String>& catalog) const{
    return getCatalog_(catalog);
  }

  bool getRecord(const std::string& name, DlcRecord& record) const{
    auto h = Snpe_DlContainer_GetRecord(handle(), name.c_str());
    if(!h) return false;
    DlcRecordInternal internal(moveHandle(h));
    auto data = internal.getData();

    record.name.assign(internal.getName());
    record.data.assign(data, data+internal.size());
    return true;
  }

  bool getRecord(const DlSystem::String& name, DlcRecord& record) const{
    return getRecord(static_cast<const std::string&>(name), record);
  }

  bool save(const std::string& filename){
    return Snpe_DlContainer_Save(handle(), filename.c_str());
  }

  bool save(const DlSystem::String& filename){
    return save(static_cast<const std::string&>(filename));
  }
};


} // ns DlContainer

ALIAS_IN_ZDL_NAMESPACE(DlContainer, DlcRecord)
ALIAS_IN_ZDL_NAMESPACE(DlContainer, IDlContainer)
