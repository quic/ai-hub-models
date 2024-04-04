// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 */

#ifndef _DIAGLOG_OPTIONS_H_
#define _DIAGLOG_OPTIONS_H_

#include <stdint.h>

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE Options handle
 */
typedef void* Snpe_Options_Handle_t;


SNPE_API
Snpe_Options_Handle_t Snpe_Options_Create();

/**
 * Destroys/frees a Options
 *
 * @param[in] handle : Handle to access Options object
 * @return indication of success/failures
 */
SNPE_API
Snpe_ErrorCode_t Snpe_Options_Delete(Snpe_Options_Handle_t handle);

/**
 * Gets DiagLogMask
 * diagLogMask: Enables diag logging only on the specified area mask
 *
 * @param[in] handle : Handle to access Options object
 * @return diagLogMask as a const char*
 */
SNPE_API
const char* Snpe_Options_GetDiagLogMask(Snpe_Options_Handle_t handle);

/**
 * Sets DiagLogMask
 * diagLogMask: Enables diag logging only on the specified area mask
 *
 * @param[in] handle : Handle to access Options object
 * @param[in] diagLogMask : specific area where logging needs to be enabed
 */
SNPE_API
void Snpe_Options_SetDiagLogMask(Snpe_Options_Handle_t handle, const char* diagLogMask);

/**
 * Gets logFileDirectory
 * logFileDirectory: The path to the directory where log files will be written.
 * The path may be relative or absolute. Relative paths are interpreted
 *
 * @param[in] handle : Handle to access Options object
 * @return logFileDirectory as a const char*
 */
SNPE_API
const char* Snpe_Options_GetLogFileDirectory(Snpe_Options_Handle_t handle);

/**
 * Sets logFileDirectory
 * logFileDirectory: The path to the directory where log files will be written.
 * The path may be relative or absolute. Relative paths are interpreted
 *
 * @param[in] handle : Handle to access Options object
 * @param[in] logFileDirectory : path for saving the log files
 */
SNPE_API
void Snpe_Options_SetLogFileDirectory(Snpe_Options_Handle_t handle, const char* logFileDirectory);


/**
 * Gets logFileName
 * logFileName: The name used for log files. If this value is empty then BaseName will be
 * used as the default file name.
 *
 * @param[in] handle : Handle to access Options object
 * @return logFileName as a const char*
 */
SNPE_API
const char* Snpe_Options_GetLogFileName(Snpe_Options_Handle_t handle);

/**
 * Sets logFileName
 * logFileName: The name used for log files. If this value is empty then BaseName will be
 * used as the default file name.
 *
 * @param[in] handle : Handle to access Options object
 * @param[in] logFileName : name of log file
 */
SNPE_API
void Snpe_Options_SetLogFileName(Snpe_Options_Handle_t handle, const char* logFileName);

/**
 * Gets the maximum number of log files to create. If set to 0 no log rotation
 * will be used and the log file name specified will be used each time, overwriting
 * any existing log file that may exist.
 *
 * @param[in] handle : Handle to access options object.
 * @return max log files to create
 */
SNPE_API
uint32_t Snpe_Options_GetLogFileRotateCount(Snpe_Options_Handle_t handle);

/**
 * Sets the maximum number of log files to create. If set to 0 no log rotation
 * will be used and the log file name specified will be used each time, overwriting
 * any existing log file that may exist.
 *
 * @param[in] handle : Handle to access options object.
 * @param[in] logFileRotateCount : max log files to create
 */
SNPE_API
void Snpe_Options_SetLogFileRotateCount(Snpe_Options_Handle_t handle, uint32_t logFileRotateCount);

/**
 * If the log file already exists, control whether it will be replaced
 *
 * @param[in] handle : Handle to access options object
 * @return 1 if log file will be replaced, 0 otherwise
 */
SNPE_API
int Snpe_Options_GetLogFileReplace(Snpe_Options_Handle_t handle);

/**
 * If the log file already exists, control whether it will be replaced
 *
 * @param[in] handle : Handle to access options object
 * @param[in] logFileReplace : 1 if log file to be replaced, 0 otherwise
 */
SNPE_API
void Snpe_Options_SetLogFileReplace(Snpe_Options_Handle_t handle, int logFileReplace);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _DIAGLOG_OPTIONS_H_
