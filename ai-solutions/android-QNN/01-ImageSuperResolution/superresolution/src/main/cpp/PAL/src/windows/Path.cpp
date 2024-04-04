//==============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <shlwapi.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "Common.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

//------------------------------------------------------------------------------
//    PAL::Path::GetSeparator
//------------------------------------------------------------------------------
char pal::Path::getSeparator() { return '/'; }

//------------------------------------------------------------------------------
//    pal::Path::Combine
//------------------------------------------------------------------------------
std::string pal::Path::combine(const std::string &s1, const std::string &s2) {
  std::stringstream ss;
  ss << s1;
  if (s1.size() > 0 && ((s1[s1.size() - 1] != '/') && (s1[s1.size() - 1] != '\\'))) {
    ss << getSeparator();
  }
  ss << s2;
  return ss.str();
}

//------------------------------------------------------------------------------
//    pal::Path::getDirectoryName
//------------------------------------------------------------------------------
std::string pal::Path::getDirectoryName(const std::string &path) {
  std::string rc = path;
  int32_t index  = std::max(static_cast<int32_t>(path.find_last_of('\\')),
                           static_cast<int32_t>(path.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = path.substr(0, index);
  }
  pal::normalizeSeparator(rc);
  return rc;
}

//------------------------------------------------------------------------------
//    pal::Path::getAbsolute
//------------------------------------------------------------------------------
std::string pal::Path::getAbsolute(const std::string &path) {
  std::string res = pal::FileOp::getAbsolutePath(path);
  pal::normalizeSeparator(res);
  return res;
}

//------------------------------------------------------------------------------
//    PAL::Path::isAbsolute
//    requirement : shlwapi.lib
//------------------------------------------------------------------------------
bool pal::Path::isAbsolute(const std::string &path) {
  std::string windowsPath = path;
  // in windows, when we need to check relative or absolute path,
  // separator MUST be '\\' rather than '/'
  // for more information : https://docs.microsoft.com/en-us/dotnet/standard/io/file-path-formats
  replace(windowsPath.begin(), windowsPath.end(), '/', '\\');
  return PathIsRelativeA(windowsPath.c_str()) == false;
}
