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

#ifndef DL_SYSTEM_IOBUFFER_DATATYPE_MAP_H
#define DL_SYSTEM_IOBUFFER_DATATYPE_MAP_H

#include <stddef.h>

#include "DlSystem/DlError.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE IOBufferDataTypeMap handle
 */
typedef void* Snpe_IOBufferDataTypeMap_Handle_t;

/**
 * @brief .
 *
 * Creates a new Buffer Data type map
 *
 */
SNPE_API
Snpe_IOBufferDataTypeMap_Handle_t Snpe_IOBufferDataTypeMap_Create();

/**
 * @brief Destroys the map
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IOBufferDataTypeMap_Delete(Snpe_IOBufferDataTypeMap_Handle_t handle);
/**
 * @brief Adds a name and the corresponding buffer data type
 *        to the map
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @param[in] name : The name of the buffer
 *
 * @param[in] bufferDataType : data type of the buffer
 *
 * @note If a buffer with the same name already exists, no new
 *       buffer is added.
 */
SNPE_API
Snpe_ErrorCode_t
Snpe_IOBufferDataTypeMap_Add(Snpe_IOBufferDataTypeMap_Handle_t handle, const char* name, Snpe_IOBufferDataType_t bufferDataType);

/**
 * @brief Removes a buffer name from the map
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @param[in] name : The name of the buffer
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IOBufferDataTypeMap_Remove(Snpe_IOBufferDataTypeMap_Handle_t handle, const char* name);

/**
 * @brief Returns the type of the named buffer
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @param[in] name : The name of the buffer
 *
 * @return The type of the buffer, or UNSPECIFIED if the buffer does not exist
 *
 */
SNPE_API
Snpe_IOBufferDataType_t Snpe_IOBufferDataTypeMap_GetBufferDataType(Snpe_IOBufferDataTypeMap_Handle_t handle, const char* name);

/**
 * @brief Returns the type of the first buffer
 *
 * @param handle : Handle to access the IOBufferDataType map
 *
 * @return The type of the first buffer, or SNPE_IO_BUFFER_DATATYPE_UNSPECIFIED if the map is empty.
 */
SNPE_API
Snpe_IOBufferDataType_t Snpe_IOBufferDataTypeMap_GetBufferDataTypeOfFirst(Snpe_IOBufferDataTypeMap_Handle_t handle);

/**
 * @brief Returns the size of the buffer type map.
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @return The size of the map
 *
 */
SNPE_API
size_t Snpe_IOBufferDataTypeMap_Size(Snpe_IOBufferDataTypeMap_Handle_t handle);

/**
 * @brief Checks the existence of the named buffer in the map
 *
 * @param[in] handle : Handle to access the IOBufferDataType map
 *
 * @param[in] name : The name of the buffer
 *
 * @return 1 if the named buffer exists, 0 otherwise.
 *
 */
SNPE_API
int Snpe_IOBufferDataTypeMap_Find(Snpe_IOBufferDataTypeMap_Handle_t handle, const char* name);

/**
 * @brief Resets the map
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IOBufferDataTypeMap_Clear(Snpe_IOBufferDataTypeMap_Handle_t handle);

/**
 * @brief Checks whether the map is empty
 *
 * @return 1 if the map is empty, 0 otherwise.
 *
 */
SNPE_API
int Snpe_IOBufferDataTypeMap_Empty(Snpe_IOBufferDataTypeMap_Handle_t handle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_IOBUFFER_DATATYPE_MAP_H
