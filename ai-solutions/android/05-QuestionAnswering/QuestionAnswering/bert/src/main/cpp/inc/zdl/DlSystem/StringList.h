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

#ifndef DL_SYSTEM_STRING_LIST_H
#define DL_SYSTEM_STRING_LIST_H

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A typedef to indicate a SNPE StringList handle
 */
typedef void* Snpe_StringList_Handle_t;

/**
 * Constructs a StringList and returns a handle to it
 *
 * @return the handle to the created StringList
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_StringList_Create();

/**
 * Constructs a StringList and returns a handle to it
 *
 * @param[in] size : size of list
 *
 * @return the handle to the created StringList
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_StringList_CreateSize(size_t size);

/**
 * Constructs a StringList and returns a handle to it
 *
 * @param[in] other : StringList handle to be copied from
 *
 * @return the handle to the created StringList
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_StringList_CreateCopy(Snpe_StringList_Handle_t other);

/**
 * Destroys/frees a StringList
 *
 * @param[in] stringListHandle : Handle to access the stringList
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_StringList_Delete(Snpe_StringList_Handle_t stringListHandle);


/**
 * Append a string to the list.
 *
 * @param[in] stringListHandle : Handle to access the stringList
 * @param[in] str Null-terminated ASCII string to append to the list.
 *
 * @return SNPE_SUCCESS if Append operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_StringList_Append(Snpe_StringList_Handle_t stringListHandle, const char* string);

/**
 * Returns the string at the indicated position,
 *  or an empty string if the positions is greater than the size
 *  of the list.
 *
 * @param[in] stringListHandle : Handle to access the stringList
 * @param[in] idx Position in the list of the desired string
 *
 * @return the string at the indicated position
 */
SNPE_API
const char* Snpe_StringList_At(Snpe_StringList_Handle_t stringListHandle, size_t idx);

/**
 * Pointer to the first string in the list.
 *  Can be used to iterate through the list.
 *
 * @param[in] stringListHandle : Handle to access the stringList
 *
 * @return Pointer to the first string in the list.
 */
SNPE_API
const char** Snpe_StringList_Begin(Snpe_StringList_Handle_t stringListHandle);

/**
 * Pointer to one after the last string in the list.
 *  Can be used to iterate through the list.
 *
 * @param[in] stringListHandle : Handle to access the stringList
 *
 * @return Pointer to one after the last string in the list
 */
SNPE_API
const char** Snpe_StringList_End(Snpe_StringList_Handle_t stringListHandle);

/**
 * Return the number of valid string pointers held by this list.
 *
 * @param[in] stringListHandle : Handle to access the stringList
 *
 * @return The size of the StringList
 */
SNPE_API
size_t Snpe_StringList_Size(Snpe_StringList_Handle_t stringListHandle);

/**
 * Copy-assigns the contents of src into dst
 *
 * @param src Source StringList handle
 * @param dst Destination StringList handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_StringList_Assign(Snpe_StringList_Handle_t src, Snpe_StringList_Handle_t dst);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_STRING_LIST_H
