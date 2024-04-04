//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once


#include <string>


#include "Wrapper.hpp"

namespace DlSystem{


// Just a backwards compatible wrapper for std::string
class String{
public:
  String() = delete;
  explicit String(const std::string& str)
    : m_String(str)
  {  }
  explicit String(std::string&& str) noexcept
    : m_String(std::move(str))
  {  }

  explicit String(const char* str)
    : m_String(str)
  {  }

  String(String&& other) noexcept = default;
  String(const String& other) = delete;


  String& operator=(String&& other) noexcept = default;
  String& operator=(const String& other) = delete;

  bool operator<(const String& rhs) const noexcept{ return m_String < rhs.m_String; }
  bool operator>(const String& rhs) const noexcept{ return m_String > rhs.m_String; }
  bool operator<=(const String& rhs) const noexcept{ return m_String <= rhs.m_String; }
  bool operator>=(const String& rhs) const noexcept{ return m_String >= rhs.m_String; }
  bool operator==(const String& rhs) const noexcept{ return m_String == rhs.m_String; }
  bool operator!=(const String& rhs) const noexcept{ return m_String != rhs.m_String; }


  bool operator<(const std::string& rhs) const noexcept{ return m_String < rhs; }
  bool operator>(const std::string& rhs) const noexcept{ return m_String > rhs; }
  bool operator<=(const std::string& rhs) const noexcept{ return m_String <= rhs; }
  bool operator>=(const std::string& rhs) const noexcept{ return m_String >= rhs; }
  bool operator==(const std::string& rhs) const noexcept{ return m_String == rhs; }
  bool operator!=(const std::string& rhs) const noexcept{ return m_String != rhs; }


  const char* c_str() const noexcept{ return m_String.c_str(); }

  explicit operator std::string&() noexcept{ return m_String; }
  explicit operator const std::string&() const noexcept{ return m_String; }

private:
  std::string m_String;
};


} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, String)
