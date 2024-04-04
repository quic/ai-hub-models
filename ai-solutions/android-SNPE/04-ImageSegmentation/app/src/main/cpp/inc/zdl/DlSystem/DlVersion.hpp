//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <string>
#include <cstdint>

#include "Wrapper.hpp"
#include "String.hpp"

#include "DlSystem/DlVersion.h"
#include "SNPE/SNPEUtil.h"


namespace DlSystem {

class  Version_t : public Wrapper<Version_t, Snpe_DlVersion_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_DlVersion_Delete};

  template<typename MemberType>
  using MajorReference = WrapperDetail::GenericConstMemberReference<Version_t, HandleType, MemberType, Snpe_DlVersion_GetMajor>;

  template<typename MemberType>
  using MinorReference = WrapperDetail::GenericConstMemberReference<Version_t, HandleType, MemberType, Snpe_DlVersion_GetMinor>;

  template<typename MemberType>
  using TeenyReference = WrapperDetail::GenericConstMemberReference<Version_t, HandleType, MemberType, Snpe_DlVersion_GetTeeny>;


  static std::string BuildGetter(Snpe_DlVersion_Handle_t handle){
    return Snpe_DlVersion_GetBuild(handle);
  }

  template<typename MemberType>
  using BuildReference = WrapperDetail::GenericConstMemberReference<Version_t, HandleType, MemberType, BuildGetter>;


  static const std::string& toString(int32_t Major, int32_t Minor, int32_t Teeny, const std::string& Build){
    thread_local std::string toret;

    toret = std::to_string(Major);
    toret += '.';
    toret += std::to_string(Minor);
    toret += '.';
    toret += std::to_string(Teeny);
    if(!Build.empty()){
      toret += '.';
      toret += Build;
    }

    return toret;
  }

public:
  Version_t()
    : BaseType(Snpe_DlVersion_Create())
  {  }

  Version_t(int32_t Major, int32_t Minor, int32_t Teeny, const std::string& Build)
    : BaseType(Snpe_DlVersion_FromString(toString(Major, Minor, Teeny, Build).c_str()))
  {  }


  /// Holds the major version number. Changes in this value indicate
  /// major changes that break backward compatibility.
  MajorReference<int32_t>         Major{*this};

  /// Holds the minor version number. Changes in this value indicate
  /// minor changes made to library that are backwards compatible
  /// (such as additions to the interface).
  MinorReference<int32_t>         Minor{*this};

  /// Holds the teeny version number. Changes in this value indicate
  /// changes such as bug fixes and patches made to the library that
  /// do not affect the interface.
  TeenyReference<int32_t>         Teeny{*this};

  /// This string holds information about the build version.
  BuildReference<std::string>     Build{*this};


  static Version_t fromString(const std::string& stringValue){
    return moveHandle(Snpe_DlVersion_FromString(stringValue.c_str()));
  }

  /**
   * @brief Returns a string in the form Major.Minor.Teeny.Build
   *
   * @return A formatted string holding the version information.
   */
  std::string toString() const{
    return Snpe_DlVersion_ToString(handle());
  }

  /**
   * @brief Returns a string in the form Major.Minor.Teeny.Build
   *
   * @return A formatted string holding the version information.
   */
  String asString() const{
    return String(toString());
  }
};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, Version_t)
