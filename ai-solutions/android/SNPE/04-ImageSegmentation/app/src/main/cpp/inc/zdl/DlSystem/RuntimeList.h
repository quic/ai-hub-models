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

#ifndef DL_SYSTEM_RUNTIME_LIST_H
#define DL_SYSTEM_RUNTIME_LIST_H

#include <stddef.h>

#include "DlSystem/DlEnums.h"
#include "DlSystem/DlError.h"

#include "StringList.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
   * A typedef to indicate a SNPE RuntimeList handle
 */
typedef void* Snpe_RuntimeList_Handle_t;

/**
 * @brief .
 *
 * Creates a new runtime list
 *
 */
SNPE_API
Snpe_RuntimeList_Handle_t Snpe_RuntimeList_Create();


/**
 * Copy-Constructs a RuntimeList and returns a handle to it
 *
 * @param runtimeListHandle the other RuntimeList to copy
 *
 * @return the handle to the created RuntimeList
 */
SNPE_API
Snpe_RuntimeList_Handle_t Snpe_RuntimeList_CreateCopy(Snpe_RuntimeList_Handle_t runtimeListHandle);

/**
 * @brief Destroys the RuntimeList
 *
 * @param[in] runtimeListHandle : Handle needed to access the runtimeList
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_Delete(Snpe_RuntimeList_Handle_t runtimeListHandle);

/**
 * Copy-assigns the contents of srcHandle into dstHandle
 *
 * @param src Source RuntimeList handle
 *
 * @param dst Destination RuntimeList handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_Assign(Snpe_RuntimeList_Handle_t src, Snpe_RuntimeList_Handle_t dst);

/**
 * @brief Returns the Runtime from list at position index
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @param[in] index : position in runtimeList
 *
 * @return The Runtime from list at position index
 */
SNPE_API
Snpe_Runtime_t Snpe_RuntimeList_GetRuntime(Snpe_RuntimeList_Handle_t runtimeListHandle, int index);

/**
 * @brief Set the Runtime of the list at position index
 *
 * @param[in] runtimeListHandle : Handle needed to access the runtimeList
 *
 * @param[in] index : position in runtimeList
 *
 * @param[in] runtime : The Runtime to assign to position index
 *
 * @return SNPE_SUCCESS on success
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_SetRuntime(Snpe_RuntimeList_Handle_t runtimeListHandle, size_t index, Snpe_Runtime_t runtime);

/**
 * @brief Adds runtime to the end of the runtime list
 *        order of precedence is former followed by latter entry
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @param[in] runtime to add
 *
 * @return Error code. Ruturns SNPE_SUCCESS If the runtime added successfully
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_Add(Snpe_RuntimeList_Handle_t runtimeListHandle, Snpe_Runtime_t runtime);

/**
 * @brief Removes the runtime from the list
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @param[in] runtime to be removed
 *
 * @return Error code. Ruturns SNPE_SUCCESS If the runtime removed successfully
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_Remove(Snpe_RuntimeList_Handle_t runtimeListHandle, Snpe_Runtime_t runtime) ;

/**
 * @brief Returns the number of runtimes in the list
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @return number of entries in the runtimeList.
 */
SNPE_API
size_t Snpe_RuntimeList_Size(Snpe_RuntimeList_Handle_t runtimeListHandle) ;

/**
 * @brief Returns 1 if the list is empty
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @return 1 if list empty, 0 otherwise.
 */
SNPE_API
int Snpe_RuntimeList_Empty(Snpe_RuntimeList_Handle_t runtimeListHandle) ;

/**
 * @brief .
 *
 * Removes all runtime from the list
 *
 * @param[in] runtimeListHandle: Handle needed to access the runtimeList
 *
 * @return Error code. Returns SNPE_SUCCESS if runtime list is cleared successfully.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeList_Clear(Snpe_RuntimeList_Handle_t runtimeListHandle);

/**
 * @brief Get a StringList of names from the runtime list in order of precedence
 *
 * @param runtimeListHandle Handle to a RuntimeList
 *
 * @return Handle to a StringList
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_RuntimeList_GetRuntimeListNames(Snpe_RuntimeList_Handle_t runtimeListHandle);

/**
 * @brief .
 *
 * @param[in] runtime const char*
 * Returns a Runtime enum corresponding to the in param string
 *
 */
SNPE_API
Snpe_Runtime_t Snpe_RuntimeList_StringToRuntime(const char* str);

/**
 * @brief .
 *
 * @param[in] runtime
 * Returns a const char* corresponding to the in param runtime enum
 *
 */
SNPE_API
const char* Snpe_RuntimeList_RuntimeToString(Snpe_Runtime_t runtime);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_RUNTIME_LIST_H
