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

#ifndef _DIAGLOG_IDIAGLOG_H_
#define _DIAGLOG_IDIAGLOG_H_

#include "DiagLog/Options.h"
#include "DlSystem/SnpeApiExportDefine.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE IDiagLog handle
 */
typedef void* Snpe_IDiagLog_Handle_t;

/**
 * @brief .
 *
 * Sets the options after initialization occurs.
 *
 * @param[in] handle : Handle to access IDiagLog
 * @param[in] loggingOptions : The options to set up diagnostic logging.
 *
 * @return Error code if the options could not be set. Ensure logging is not started/
 *         SNPE_SUCCESS otherwise
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IDiagLog_SetOptions(Snpe_IDiagLog_Handle_t handle, Snpe_Options_Handle_t loggingOptionsHandle);

/**
 * @brief .
 *
 * Gets the curent options for the diag logger.
 *
 * @param[in] handle : Handle to access IDiagLog
 * @return Handle to access DiagLog options.
 */
SNPE_API
Snpe_Options_Handle_t Snpe_IDiagLog_GetOptions(Snpe_IDiagLog_Handle_t handle);

/**
 * @brief .
 *
 * @param[in] handle : Handle to access IDiagLog
 * @param[in] mask : Allows for setting the log mask once diag logging has started
 * @return SNPE_SUCCESS if the level was set successfully.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IDiagLog_SetDiagLogMask(Snpe_IDiagLog_Handle_t handle, const char* mask) ;

/**
 * @brief .
 *
 * Enables logging.
 *
 * Logging should be started prior to the instantiation of other SNPE_APIs
 * to ensure all events are captured.
 *
 * @param[in] handle : Handle to access IDiagLog
 * @return SNPE_SUCCESS if diagnostic logging started successfully.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IDiagLog_Start(Snpe_IDiagLog_Handle_t handle);

/**
 * @brief Disables logging.
 *
 * @param[in] handle : Handle to access IDiagLog
 *
 * @return SNPE_SUCCESS if logging stopped successfully. Error code otherwise.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IDiagLog_Stop(Snpe_IDiagLog_Handle_t handle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _DIAGLOG_IDIAGLOG_H_
