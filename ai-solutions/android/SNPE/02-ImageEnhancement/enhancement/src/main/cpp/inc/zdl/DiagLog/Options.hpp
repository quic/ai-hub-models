//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"

#include <string>
#include <cstdint>

#include "DiagLog/IDiagLog.h"


namespace DiagLog {

class Options
{
public:
  Options(
    std::string diagLogMask = "",
    std::string logFileDirectory = "diaglogs",
    std::string logFileName = "DiagLog",
    uint32_t logFileRotateCount = 20,
    bool logFileReplace = true
  )
    : DiagLogMask(std::move(diagLogMask)),
      LogFileDirectory(std::move(logFileDirectory)),
      LogFileName(std::move(logFileName)),
      LogFileRotateCount(logFileRotateCount),
      LogFileReplace(logFileReplace)
  {
    // Solves the empty string problem with multiple std libs
    DiagLogMask.reserve(1);
  }

  std::string DiagLogMask;
  std::string LogFileDirectory;
  std::string LogFileName;
  uint32_t LogFileRotateCount;

  bool LogFileReplace;
};

} // ns DiagLog

ALIAS_IN_ZDL_NAMESPACE(DiagLog, Options)
