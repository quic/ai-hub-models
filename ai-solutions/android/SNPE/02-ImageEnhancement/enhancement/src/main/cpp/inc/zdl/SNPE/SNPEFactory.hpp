//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"

#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"


#include "SNPE/SNPEUtil.h"
#include "DlSystem/DlEnums.h"

namespace SNPE {


class SNPEFactory {
public:


  static bool isRuntimeAvailable(DlSystem::Runtime_t runtime){
    return Snpe_Util_IsRuntimeAvailable(static_cast<Snpe_Runtime_t>(runtime));
  }

  static bool isRuntimeAvailable(DlSystem::Runtime_t runtime, DlSystem::RuntimeCheckOption_t option){
    return Snpe_Util_IsRuntimeAvailableCheckOption(static_cast<Snpe_Runtime_t>(runtime),
                                                   static_cast<Snpe_RuntimeCheckOption_t>(option));
  }

  static DlSystem::ITensorFactory& getTensorFactory(){
    static DlSystem::ITensorFactory iTensorFactory;
    return iTensorFactory;
  }

  static DlSystem::IUserBufferFactory& getUserBufferFactory(){
    static DlSystem::IUserBufferFactory iUserBufferFactory;
    return iUserBufferFactory;
  }

  static DlSystem::Version_t getLibraryVersion(){
    return WrapperDetail::moveHandle(Snpe_Util_GetLibraryVersion());
  }

  static bool setSNPEStorageLocation(const char* storagePath){
    return SNPE_SUCCESS == Snpe_Util_SetSNPEStorageLocation(storagePath);
  }

  static bool addOpPackage(const std::string& regLibraryPath){
    return SNPE_SUCCESS == Snpe_Util_AddOpPackage(regLibraryPath.c_str());
  }

  static bool isGLCLInteropSupported(){
    return Snpe_Util_IsGLCLInteropSupported();
  }

  static const char* getLastError(){
    return Snpe_Util_GetLastError();
  }

  static bool initializeLogging(const DlSystem::LogLevel_t& level){
    return Snpe_Util_InitializeLogging(static_cast<Snpe_LogLevel_t>(level));
  }

  static bool initializeLogging(const DlSystem::LogLevel_t& level, const std::string& logPath){
    return Snpe_Util_InitializeLoggingPath(static_cast<Snpe_LogLevel_t>(level), logPath.c_str());
  }

  static bool setLogLevel(const DlSystem::LogLevel_t& level){
    return Snpe_Util_SetLogLevel(static_cast<Snpe_LogLevel_t>(level));
  }

  static bool terminateLogging(){
    return Snpe_Util_TerminateLogging();
  }
};


} // ns SNPE


ALIAS_IN_ZDL_NAMESPACE(SNPE, SNPEFactory)
