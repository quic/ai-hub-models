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

#ifndef DL_SYSTEM_USER_MEMORY_MAP_H
#define DL_SYSTEM_USER_MEMORY_MAP_H

#include "DlSystem/StringList.h"
#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE User Memory handle
 */
typedef void* Snpe_UserMemoryMap_Handle_t;

/**
 * @brief .
 *
 * Creates a new empty UserMemory map
 */
SNPE_API
Snpe_UserMemoryMap_Handle_t Snpe_UserMemoryMap_Create();

/**
 * copy constructor.
 * @param[in] other : Handle to the other object to copy.
 */
SNPE_API
Snpe_UserMemoryMap_Handle_t Snpe_UserMemoryMap_Copy(Snpe_UserMemoryMap_Handle_t other);

/**
 * Copy-assigns the contents of srcHandle into dstHandle
 *
 * @param[in] srcHandle Source UserMemoryMap handle
 *
 * @param[out] dstHandle Destination UserMemoryMap handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserMemoryMap_Assign(Snpe_UserMemoryMap_Handle_t srcHandle, Snpe_UserMemoryMap_Handle_t dstHandle);

/**
 * Destroys/frees UserMemory Map
 *
 * @param[in] handle : Handle to access UserMemory Map
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserMemoryMap_Delete(Snpe_UserMemoryMap_Handle_t handle);

/**
 * @brief Adds a name and the corresponding buffer address
 *        to the map
 *
 * @param[in] handle : Handle to access UserMemory Map
 * @param[in] name : The name of the UserMemory
 * @param[in] address : The pointer to the Buffer Memory
 *
 * @note If a UserBuffer with the same name already exists, the new
 *       address would be updated.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserMemoryMap_Add(Snpe_UserMemoryMap_Handle_t handle, const char *name, void *address);

/**
 * @brief Removes a mapping of one Buffer address and its name by its name
 *
 * @param[in] handle : Handle to access UserMemory Map
 * @param[in] name : The name of Memory address to be removed
 *
 * @note If no UserBuffer with the specified name is found, nothing
 *       is done.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserMemoryMap_Remove(Snpe_UserMemoryMap_Handle_t handle, const char *name);

/**
 * @brief Returns the number of User Memory addresses in the map
 * @param[in] handle : Handle to access UserMemory Map
 */
SNPE_API
size_t Snpe_UserMemoryMap_Size(Snpe_UserMemoryMap_Handle_t handle);

/**
 * @brief .
 *
 * Removes all User Memory from the map
 * @param[in] handle : Handle to access UserMemory Map
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserMemoryMap_Clear(Snpe_UserMemoryMap_Handle_t handle);

/**
 * @brief .
 * Returns the names of all User Memory
 *
 * @param[in] handle : Handle to access UserMemory Map
 *
 * @return Returns a handle to the stringList.
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_UserMemoryMap_GetUserBufferNames(Snpe_UserMemoryMap_Handle_t handle);

/**
 * @brief Returns the no of UserMemory addresses mapped to the buffer
 *
 * @param[in] handle : Handle to access UserMemory Map
 * @param[in] name : The name of the UserMemory
 *
 */
SNPE_API
size_t Snpe_UserMemoryMap_GetUserMemoryAddressCount(Snpe_UserMemoryMap_Handle_t handle, const char *name);

/**
 * @brief Returns address at a specified index corresponding to a UserMemory buffer name
 *
 * @param[in] handle : Handle to access UserMemory Map
 * @param[in] name : The name of the buffer
 * @param[in] index : The index in the list of addresses
 *
 */
SNPE_API
void* Snpe_UserMemoryMap_GetUserMemoryAddressAtIndex(Snpe_UserMemoryMap_Handle_t handle, const char *name, uint32_t index);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_USER_MEMORY_MAP_H
