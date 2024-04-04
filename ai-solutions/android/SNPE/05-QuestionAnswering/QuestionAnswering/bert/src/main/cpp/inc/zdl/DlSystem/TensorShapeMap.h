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

#ifndef _SNPE_TENSOR_SHAPE_MAP_H_
#define _SNPE_TENSOR_SHAPE_MAP_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"

#include "DlSystem/TensorShape.h"
#include "DlSystem/StringList.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
  * A typedef to indicate a SNPE TensorShapeMap handle
 */
typedef void* Snpe_TensorShapeMap_Handle_t;

/**
 * Constructs a TensorShapeMap and returns a handle to it
 *
 * @return the handle to the created TensorShapeMap
 */
SNPE_API
Snpe_TensorShapeMap_Handle_t Snpe_TensorShapeMap_Create();

/**
 * @brief .
 *
 * copy constructor.
 *
 * @param[in] tsmHandle : Handle to the other object to copy.
 * @return the handle to the created TensorShapeMap
 */
SNPE_API
Snpe_TensorShapeMap_Handle_t Snpe_TensorShapeMap_CreateCopy(Snpe_TensorShapeMap_Handle_t tsmHandle);

/**
 * Destroys/frees Tensor Shape Map
 *
 * @param[in] tsmhandle : handle to access Tensor Shape Map
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShapeMap_Delete(Snpe_TensorShapeMap_Handle_t tsmHandle);

/**
 * @brief .
 *
 * assignment operator. Copy-assigns from srcHandle to dstHandle
 * @param[in] srcHandle : handle to source Tensor Shape Map object
 * @param[out] dstHandle : handle to destination Tensor Shape Map object
 *
 * @return Returns SNPE_SUCCESS if Assignment successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShapeMap_Assign(Snpe_TensorShapeMap_Handle_t srcHandle, Snpe_TensorShapeMap_Handle_t dstHandle);

/**
 * @brief Adds a name and the corresponding tensor pointer
 *        to the map
 *
 * @param[in] tsmhandle : handle to access Tensor Shape Map
 * @param[in] name The name of the tensor
 * @param[in] tsHandle : Handle to access Tensor Shape
 *
 * @return Returns SNPE_SUCCESS if Add operation successful
 * @note If a tensor with the same name already exists, no new
 *       tensor is added.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShapeMap_Add(Snpe_TensorShapeMap_Handle_t tsmHandle, const char* name, Snpe_TensorShape_Handle_t tsHandle);

/**
 * @brief Removes a mapping of tensor and its name by its name
 *
 * @param[in] tsmhandle : handle to access Tensor Shape Map
 * @param[in] name The name of tensor to be removed
 * @return Returns SNPE_SUCCESS if Remove operation successful
 *
 * @note If no tensor with the specified name is found, nothing
 *       is done.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShapeMap_Remove(Snpe_TensorShapeMap_Handle_t tsmHandle, const char* name);

/**
 * @brief Returns the number of tensors in the map
 * @param[in] tsmhandle : handle to access Tensor Shape Map    
 * @return Returns number entries in TensorShapeMap
 */
SNPE_API
size_t Snpe_TensorShapeMap_Size(Snpe_TensorShapeMap_Handle_t tsmHandle);

/**
 * @brief .
 *
 * Removes all tensors from the map
 * @param[in] tsmhandle : handle to access Tensor Shape Map
 * @return Returns SNPE_SUCCESS if Clear operation successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShapeMap_Clear(Snpe_TensorShapeMap_Handle_t tsmHandle);

/**
 * @brief Returns the tensor given its name.
 *
 * @param[in] tsmhandle : handle to access Tensor Shape Map
 * @param[in] name The name of the tensor to get.
 *
 * @return nullptr if no tensor with the specified name is
 *         found; otherwise, a valid Tensor Shape Handle.
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_TensorShapeMap_GetTensorShape(Snpe_TensorShapeMap_Handle_t tsmHandle, const char* name);

/**
 * @brief .
 *
 * @param[in] tsmHandle : handle to access Tensor Shape Map
 * @return A stringList Handle to access names of all tensor shapes
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_TensorShapeMap_GetTensorShapeNames(Snpe_TensorShapeMap_Handle_t tsmHandle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_TENSOR_SHAPE_MAP_H_
