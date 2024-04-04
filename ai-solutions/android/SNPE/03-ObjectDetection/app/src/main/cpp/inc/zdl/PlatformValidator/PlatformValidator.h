// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
//==============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 */

#ifndef _PLATFORM_VALIDATOR_H_
#define _PLATFORM_VALIDATOR_H_

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"
#include "DlSystem/DlEnums.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE PlatformValidator handle
 */
typedef void* Snpe_PlatformValidator_Handle_t;

/**
 * @brief .
 *
 * Creates a new Platform Validator
 *
 */
SNPE_API
Snpe_PlatformValidator_Handle_t Snpe_PlatformValidator_Create();


/**
 * Destroys/frees Platform Validator
 *
 * @param[in] handle : Handle to access Platform Validator
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PlatformValidator_Delete(Snpe_PlatformValidator_Handle_t handle);

/**
 * @brief Sets the runtime processor for compatibility check
 *
 * @return Void
 */
SNPE_API
void Snpe_PlatformValidator_SetRuntime(Snpe_PlatformValidator_Handle_t handle,
                                       Snpe_Runtime_t runtime,
                                       bool unsignedPD=true);

/**
 * @brief Checks if the Runtime prerequisites for SNPE are available.
 *
 * @return 1 if the Runtime prerequisites are available, else 0.
 */
SNPE_API
int Snpe_PlatformValidator_IsRuntimeAvailable(Snpe_PlatformValidator_Handle_t handle,
                                              bool unsignedPD=true);

/**
 * @brief Returns the core version for the Runtime selected.
 *
 * @return char* which contains the actual core version value
 */
SNPE_API
const char* Snpe_PlatformValidator_GetCoreVersion(Snpe_PlatformValidator_Handle_t handle);

/**
 * @brief Returns the library version for the Runtime selected.
 *
 * @return char* which contains the actual lib version value
 */
SNPE_API
const char* Snpe_PlatformValidator_GetLibVersion(Snpe_PlatformValidator_Handle_t handle);

/**
 * @brief Runs a small program on the runtime and Checks if SNPE is supported for Runtime.
 *
 * @return If 1, the device is ready for SNPE execution, else return 0.
 */
SNPE_API
int Snpe_PlatformValidator_RuntimeCheck(Snpe_PlatformValidator_Handle_t handle,
                                        bool unsignedPD=true);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _PLATFORM_VALIDATOR_H_
