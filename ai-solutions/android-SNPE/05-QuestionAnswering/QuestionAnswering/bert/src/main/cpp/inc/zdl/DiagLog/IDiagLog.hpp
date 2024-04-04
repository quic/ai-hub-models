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

#include "Options.hpp"
#include "DlSystem/String.hpp"

#include "DiagLog/IDiagLog.h"


namespace DiagLog{
class IDiagLog : public Wrapper<IDiagLog, Snpe_IDiagLog_Handle_t> {
  friend BaseType;
  // Use this to get free move Ctor and move assignment operator, provided this class does not specify
  // as copy assignment operator or copy Ctor
  using BaseType::BaseType;

  static Snpe_ErrorCode_t InvalidDeleteCall(Snpe_IDiagLog_Handle_t ){
    return SNPE_ERRORCODE_CAPI_DELETE_FAILURE;
  }

  static constexpr DeleteFunctionType DeleteFunction{InvalidDeleteCall};

  class OptionsInternal : public Wrapper<OptionsInternal, Snpe_Options_Handle_t> {
    friend BaseType;
    // Use this to get free move Ctor and move assignment operator, provided this class does not specify
    // as copy assignment operator or copy Ctor
    using BaseType::BaseType;

    static constexpr DeleteFunctionType DeleteFunction{Snpe_Options_Delete};
  public:
    OptionsInternal()
      : BaseType(Snpe_Options_Create())
    {  }

    explicit OptionsInternal(const Options& options)
      : BaseType(Snpe_Options_Create())
    {
      setDiagLogMask(options.DiagLogMask.c_str());
      setLogFileDirectory(options.LogFileDirectory.c_str());
      setLogFileName(options.LogFileName.c_str());
      setLogFileRotateCount(options.LogFileRotateCount);
      setLogFileReplace(options.LogFileReplace);
    }

    const char* getDiagLogMask() const{
      return Snpe_Options_GetDiagLogMask(handle());
    }
    void  setDiagLogMask(const char* diagLogMask){
      Snpe_Options_SetDiagLogMask(handle(), diagLogMask);
    }

    const char* getLogFileDirectory() const{
      return Snpe_Options_GetLogFileDirectory(handle());
    }
    void  setLogFileDirectory(const char* logFileDirectory){
      Snpe_Options_SetLogFileDirectory(handle(), logFileDirectory);
    }

    const char* getLogFileName() const{
      return Snpe_Options_GetLogFileName(handle());
    }
    void setLogFileName(const char* logFileName){
      Snpe_Options_SetLogFileName(handle(), logFileName);
    }

    uint32_t getLogFileRotateCount() const{
      return Snpe_Options_GetLogFileRotateCount(handle());
    }
    void setLogFileRotateCount(uint32_t logFileRotateCount){
      Snpe_Options_SetLogFileRotateCount(handle(), logFileRotateCount);
    }

    bool getLogFileReplace() const{
      return Snpe_Options_GetLogFileReplace(handle());
    }
    void setLogFileReplace(bool logFileReplace){
      Snpe_Options_SetLogFileReplace(handle(), logFileReplace);
    }

    explicit operator Options() const{
      return {
        getDiagLogMask(),
        getLogFileDirectory(),
        getLogFileName(),
        getLogFileRotateCount(),
        getLogFileReplace()
      };
    }

  };



public:
  bool setOptions(const Options& loggingOptions){
    OptionsInternal optionsInternal(loggingOptions);
    return SNPE_SUCCESS == Snpe_IDiagLog_SetOptions(handle(), getHandle(optionsInternal));
  }
  Options getOptions() const{
    OptionsInternal optionsInternal(moveHandle(Snpe_IDiagLog_GetOptions(handle())));
    return Options(optionsInternal);
  }

  bool setDiagLogMask(const std::string& mask){
    return SNPE_SUCCESS == Snpe_IDiagLog_SetDiagLogMask(handle(), mask.c_str());
  }
  bool setDiagLogMask(const DlSystem::String& mask){
    return setDiagLogMask(static_cast<const std::string&>(mask));
  }

  bool start(void){
    return SNPE_SUCCESS == Snpe_IDiagLog_Start(handle());
  }
  bool stop(void){
    return SNPE_SUCCESS == Snpe_IDiagLog_Stop(handle());
  }

};

} // ns DiagLog

ALIAS_IN_ZDL_NAMESPACE(DiagLog, IDiagLog)
