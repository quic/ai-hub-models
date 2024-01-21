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

#ifndef DL_SYSTEM_TENSORMAP_H
#define DL_SYSTEM_TENSORMAP_H

#include "DlSystem/ITensor.h"
#include "DlSystem/StringList.h"
#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A typedef to indicate a SNPE Tensor Map handle
 */
typedef void* Snpe_TensorMap_Handle_t;


/**
 * Constructs a TensorMap and returns a handle to it
 *
 * @return the handle to the created TensorMap
 */
SNPE_API
Snpe_TensorMap_Handle_t Snpe_TensorMap_Create();


/**
 * Copy-Constructs a TensorMap and returns a handle to it
 *
 * @param tensorMapHandle the other TensorMap to copy
 *
 * @return the handle to the created TensorMap
 */
SNPE_API
Snpe_TensorMap_Handle_t Snpe_TensorMap_CreateCopy(Snpe_TensorMap_Handle_t tensorMapHandle);

/**
 * Copy-assigns the contents of srcHandle into dstHandle
 *
 * @param src Source TensorMap handle
 *
 * @param dst Destination TensorMap handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorMap_Assign(Snpe_TensorMap_Handle_t srcHandle, Snpe_TensorMap_Handle_t dstHandle);


/**
 * Destroys/frees Tensor Map
 *
 * @param[in] handle : handle to tensorMap
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorMap_Delete(Snpe_TensorMap_Handle_t handle);

/**
 * @brief Adds a name and the corresponding tensor pointer
 *        to the map
 *
 * @param[in] handle : Handle to tensorMap
 * @param[in] name : The name of the tensor
 * @param[in] tensorHandle : Handle to access ITensor
 *
 * @note If a tensor with the same name already exists, the
 *       tensor is replaced with the existing tensor.
 */
SNPE_API
void Snpe_TensorMap_Add(Snpe_TensorMap_Handle_t handle, const char *name, Snpe_ITensor_Handle_t tensorHandle);

/**
 * @brief Removes a mapping of tensor and its name by its name
 *
 * @param[in] handle : Handle to tensorMap
 * @param[in] name : The name of tensor to be removed
 *
 * @note If no tensor with the specified name is found, nothing
 *       is done.
 */
SNPE_API
void Snpe_TensorMap_Remove(Snpe_TensorMap_Handle_t handle, const char *name);

/**
 * @brief Returns the number of tensors in the map
 *
 * @param[in] handle : Handle to tensorMap
 *
 * @return Number of tensors in the map
 */
SNPE_API
size_t Snpe_TensorMap_Size(Snpe_TensorMap_Handle_t handle);

/**
 * @brief .
 *
 * @param[in] handle : Handle to tensorMap
 * Removes all tensors from the map
 */
SNPE_API
void Snpe_TensorMap_Clear(Snpe_TensorMap_Handle_t handle);

/**
 * @brief Returns the tensor given its name.
 *  
 * @param[in] handle : Handle to tensorMap
 * @param[in] name : The name of the tensor to get. 
 *  
 * @return nullptr if no tensor with the specified name is 
 *         found; otherwise, a valid pointer to the tensor.
 */
SNPE_API
Snpe_ITensor_Handle_t Snpe_TensorMap_GetTensor_Ref(Snpe_TensorMap_Handle_t handle, const char *name);

/**
 * @brief .
 *
 * @param[in] handle : Handle to tensorMap
 *
 * @return A StringList of the names of all tensors
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_TensorMap_GetTensorNames(Snpe_TensorMap_Handle_t handle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_TENSOR_MAP_H
