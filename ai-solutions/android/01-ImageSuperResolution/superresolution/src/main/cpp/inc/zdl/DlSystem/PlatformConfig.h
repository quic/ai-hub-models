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

#ifndef DL_SYSTEM_PLATFORMCONFIG_H
#define DL_SYSTEM_PLATFORMCONFIG_H

#include "DlSystem/DlError.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief .
 *
 * A structure OpenGL configuration
 *
 * @note When certain OpenGL context and display are provided to UserGLConfig for using
 *       GPU buffer as input directly, the user MUST ensure the particular OpenGL
 *       context and display remain vaild throughout the execution of neural network models.
 */
typedef void* Snpe_UserGLConfig_Handle_t;

/**
 * @brief .
 *
 * Creates a new userGLConfig
 *
 */
SNPE_API
Snpe_UserGLConfig_Handle_t Snpe_UserGLConfig_Create();

/**
 * @brief Destroys the userGLConfig
 *
 * @param[in] handle : Handle to access the userGLConfig
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserGLConfig_Delete(Snpe_UserGLConfig_Handle_t handle);

/**
 * @brief Sets the EGL context
 *
 * @param[in] handle : Handle to access userGLConfig
 *
 * @param[in] userGLContext : void pointer
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserGLConfig_SetUserGLContext(Snpe_UserGLConfig_Handle_t handle, void* userGLContext);

/**
 * @brief Sets the EGL Display
 *
 * @param[in] handle : Handle to access userGLConfig
 *
 * @param[in] userGLDisplay : void pointer
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserGLConfig_SetUserGLDisplay(Snpe_UserGLConfig_Handle_t handle, void* userGLDisplay);


/**
 * @brief Get EGL context
 *
 * @param[in] handle : Handle to access userGLConfig
 *
 * @return userGLContext of type void pointer
 *
 */
SNPE_API
void* Snpe_UserGLConfig_GetUserGLContext(Snpe_UserGLConfig_Handle_t handle);

/**
 * @brief Get EGL Display
 *
 * @param[in] handle : Handle to access userGLConfig
 *
 * @return userGLDisplay of type void pointer
 *
 */
SNPE_API
void* Snpe_UserGLConfig_GetUserGLDisplay(Snpe_UserGLConfig_Handle_t handle);


/**
 * @brief .
 *
 * A structure Gpu configuration
 */
typedef void* Snpe_UserGpuConfig_Handle_t;

/**
 * @brief .
 *
 * Creates a new userGpuConfig
 *
 */
SNPE_API
Snpe_UserGpuConfig_Handle_t Snpe_UserGpuConfig_Create();

/**
 * @brief Destroys the userGpuConfig
 *
 * @param[in] handle : Handle to access userGLConfig
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserGpuConfig_Delete(Snpe_UserGpuConfig_Handle_t handle);

/**
 * @brief Set the userGpuConfig
 *
 * @param[in] handle : Handle to access userGpuConfig
 *
 * @param[in] glHandle : Handle needed to access userGlConfig
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
void Snpe_UserGpuConfig_Set(Snpe_UserGpuConfig_Handle_t handle, Snpe_UserGLConfig_Handle_t glHandle);

/**
 * @brief Get the userGpuConfig
 *
 * @param[in] handle : Handle to access userGpuConfig
 *
 * @return Handle needed to access userGlConfig
 */
SNPE_API
Snpe_UserGLConfig_Handle_t Snpe_UserGpuConfig_Get_Ref(Snpe_UserGpuConfig_Handle_t handle);



/**
 * A typedef to indicate a SNPE PlatformConfig handle
 */
typedef void* Snpe_PlatformConfig_Handle_t;


/**
 * @brief .
 *
 * Creates a new PlatformConfig
 *
 */
SNPE_API
Snpe_PlatformConfig_Handle_t Snpe_PlatformConfig_Create();


/**
 * @brief Copy-Construct a PlatformConfig from another PlatformConfig
 *
 * @param[in] otherHandle Handle to the other PlatformConfig
 *
 * @return Handle to the Copy-Constructed PlatformConfig
 */
SNPE_API
Snpe_PlatformConfig_Handle_t Snpe_PlatformConfig_CreateCopy(Snpe_PlatformConfig_Handle_t otherHandle);

/**
 * @brief Destroys the PlatformConfig
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PlatformConfig_Delete(Snpe_PlatformConfig_Handle_t handle);


typedef enum
{
  /// Unknown platform type.
  SNPE_PLATFORMCONFIG_PLATFORMTYPE_UNKNOWN = 0,

  /// Snapdragon CPU.
  SNPE_PLATFORMCONFIG_PLATFORMTYPE_CPU = 1,

  /// Adreno GPU.
  SNPE_PLATFORMCONFIG_PLATFORMTYPE_GPU = 2,

  /// Hexagon DSP.
  SNPE_PLATFORMCONFIG_PLATFORMTYPE_DSP = 3
} Snpe_PlatformConfig_PlatformType_t;


/**
 * @brief Retrieves the platform type
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return Platform type
 */
SNPE_API
Snpe_PlatformConfig_PlatformType_t Snpe_PlatformConfig_GetPlatformType(Snpe_PlatformConfig_Handle_t handle);

/**
 * @brief Indicates whther the plaform configuration is valid.
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return 1 if the platform configuration is valid; 0 otherwise.
 */
SNPE_API
int Snpe_PlatformConfig_IsValid(Snpe_PlatformConfig_Handle_t handle);

/**
 * @brief Retrieves the Gpu configuration
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return userGpuConfig populated with the Gpu configuration.
 *
 */
SNPE_API
Snpe_UserGpuConfig_Handle_t Snpe_PlatformConfig_GetUserGpuConfig(Snpe_PlatformConfig_Handle_t handle);

/**
 * @brief Sets the Gpu configuration
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @param[in] gpuHandle : Gpu Configuration handle
 *
 * @return 1 if Gpu configuration was successfully set; 0 otherwise.
 */
SNPE_API
int Snpe_PlatformConfig_SetUserGpuConfig(Snpe_PlatformConfig_Handle_t handle, Snpe_UserGpuConfig_Handle_t gpuHandle);

/**
 * @brief Sets the platform options
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @param[in] options : Options as a const char* in the form of "keyword:options"
 *
 * @return 1 if options are pass validation; otherwise 0.  If false, the options are not updated.
 */
SNPE_API
int Snpe_PlatformConfig_SetPlatformOptions(Snpe_PlatformConfig_Handle_t handle, const char* options);

/**
 * @brief Indicates whther the plaform configuration is valid.
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return 1 if the platform configuration is valid; 0 otherwise.
 */
SNPE_API
int Snpe_PlatformConfig_IsOptionsValid(Snpe_PlatformConfig_Handle_t handle);

/**
 * @brief Gets the platform options
 *
 * @param[in] handle : Handle needed to access the platformConfig
 *
 * @return Options as a const char*
 */
SNPE_API
const char* Snpe_PlatformConfig_GetPlatformOptions(Snpe_PlatformConfig_Handle_t handle);

/**
 * @brief Sets the platform options
 *
 * @note the returned string will be invalidated by subsequent calls to this function
 *
 * @param[in] handle : Handle needed to access the platformConfig
 * @param[in] optionName : Name of platform options"
 * @param[in] value : Value of specified optionName
 *
 * @return If 1, add "optionName:value" to platform options if optionName don't exist, otherwise update the
 *         value of specified optionName.
 *         If 0, the platform options will not be changed.
 */
SNPE_API
int Snpe_PlatformConfig_SetPlatformOptionValue(Snpe_PlatformConfig_Handle_t handle, const char* optionName, const char* value);

/**
 * @brief Removes the platform options
 *
 * @param[in] handle : Handle needed to access the platformConfig
 * @param[in] optionName : Name of platform options"
 * @param[in] value : Value of specified optionName
 *
 * @return If 1, removed "optionName:value" to platform options if optionName don't exist, do nothing.
 *         If 0, the platform options will not be changed.
 */
SNPE_API
int Snpe_PlatformConfig_RemovePlatformOptionValue(Snpe_PlatformConfig_Handle_t handle, const char* optionName, const char* value);

SNPE_API
void Snpe_PlatformConfig_SetIsUserGLBuffer(int isUserGLBuffer);

SNPE_API
int Snpe_PlatformConfig_GetIsUserGLBuffer();


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_PLATFORMCONFIG_H
