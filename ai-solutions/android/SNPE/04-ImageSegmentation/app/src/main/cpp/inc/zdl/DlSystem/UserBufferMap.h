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

#ifndef DL_SYSTEM_USER_BUFFER_MAP_H
#define DL_SYSTEM_USER_BUFFER_MAP_H

#include "DlSystem/StringList.h"
#include "DlSystem/IUserBuffer.h"
#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE UserBufferMap handle
 */
typedef void* Snpe_UserBufferMap_Handle_t;

/**
 * @brief .
 *
 * Creates a new empty UserBuffer map
 */
SNPE_API
Snpe_UserBufferMap_Handle_t Snpe_UserBufferMap_Create();

/**
 * copy constructor.
 * @param[in] other : Handle to the other userBufferMap to be copied from.
 */
SNPE_API
Snpe_UserBufferMap_Handle_t Snpe_UserBufferMap_CreateCopy(Snpe_UserBufferMap_Handle_t other);


/**
 * @brief Adds a name and the corresponding UserBuffer pointer
 *        to the map
 *
 * @param[in] handle : Handle to access UserBufferMap
 * @param[in] name : The name of the UserBuffer
 * @param[in] bufferHandle : Handle to access UserBuffer
 *
 * @note If a UserBuffer with the same name already exists, the new
 *       UserBuffer pointer would be updated.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferMap_Add(Snpe_UserBufferMap_Handle_t handle, const char *name, Snpe_IUserBuffer_Handle_t bufferHandle);

/**
 * @brief Removes a mapping of one UserBuffer and its name by its name
 *
 * @param[in] handle : Handle to access UserBufferMap
 *
 * @param[in] name : The name of UserBuffer to be removed
 *
 * @note If no UserBuffer with the specified name is found, nothing
 *       is done.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferMap_Remove(Snpe_UserBufferMap_Handle_t handle, const char *name);

/**
 * @brief Returns the number of UserBuffers in the map
 * @param[in] handle : Handle to access UserBufferMap
 */
SNPE_API
size_t Snpe_UserBufferMap_Size(Snpe_UserBufferMap_Handle_t handle);

/**
 * @brief .
 *
 * @param[in] handle : Handle to access UserBufferMap
 * Removes all UserBuffers from the map
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferMap_Clear(Snpe_UserBufferMap_Handle_t handle);

/**
 * @brief Returns the UserBuffer given its name.
 *
 * @param[in] handle : Handle to access UserBufferMap
 *
 * @param[in] name : The name of the UserBuffer to get.
 *
 * @return nullptr if no UserBuffer with the specified name is
 *         found; otherwise, a valid pointer to the UserBuffer.
 */
SNPE_API
Snpe_IUserBuffer_Handle_t Snpe_UserBufferMap_GetUserBuffer_Ref(Snpe_UserBufferMap_Handle_t handle , const char *name);

/**
 * @brief .
 *
 * Returns the names of all UserBuffers
 *
 * @param[in] handle : Handle to access UserBufferMap
 *
 * @return A list of UserBuffer names.
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_UserBufferMap_GetUserBufferNames(Snpe_UserBufferMap_Handle_t handle);

/**
 * Copy-assigns the contents of srcHandle into dstHandle
 *
 * @param src Source UserBufferMap handle
 * @param dst Destination UserBufferMap handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferMap_Assign(Snpe_UserBufferMap_Handle_t srcHandle, Snpe_UserBufferMap_Handle_t dstHandle);

/**
 * Destroys/frees UserBuffer Map
 *
 * @param[in] handle : Handle to access UserBuffer Map
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferMap_Delete(Snpe_UserBufferMap_Handle_t handle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_USER_BUFFER_MAP_H
