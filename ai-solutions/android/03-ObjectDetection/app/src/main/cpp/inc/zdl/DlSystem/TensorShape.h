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

#ifndef DL_SYSTEM_TENSOR_SHAPE_H
#define DL_SYSTEM_TENSOR_SHAPE_H

#include <stddef.h>

#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
  * A typedef to indicate a SNPE TensorShape handle
 */
typedef void* Snpe_TensorShape_Handle_t;


/**
 * @brief .
 *
 * Creates a new shape with a list of dims specified in array
 *
 * @param[in] dims The dimensions are specified in which the last
 * element of the vector represents the fastest varying
 * dimension and the zeroth element represents the slowest
 * varying, etc.
 *
 * @param[in] size Size of the array.
 *
 * @return the handle to the created TensorShape
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_TensorShape_CreateDimsSize(const size_t *dims, size_t size);

/**
 * Constructs a TensorShape and returns a handle to it
 *
 * @return the handle to the created TensorShape
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_TensorShape_Create();

/**
 * @brief .
 *
 * copy constructor.
 * @param[in] other object to copy.
 *
 * @return the handle to the created TensorShape.
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_TensorShape_CreateCopy(Snpe_TensorShape_Handle_t other);

/**
 * Destroys/frees Tensor Shape
 *
 * @param[in] handle : handle to tensorShape
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShape_Delete(Snpe_TensorShape_Handle_t tensorShapeHandle);

/**
 * Copy-assigns the contents of srcHandle into dstHandle
 *
 * @param srcHandle Source TensorShape handle
 * @param dstHandle Destination TensorShape handle
 *
 * @return SNPE_SUCCESS on successful copy-assignment
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShape_Assign(Snpe_TensorShape_Handle_t srcHandle, Snpe_TensorShape_Handle_t dstHandle);

/**
 * @brief .
 *
 * Concatenates additional dimensions specified in
 * the array to the existing dimensions.
 *
 * @param[in] handle : handle to tensorShape
 * @param[in] dims The dimensions are specified in which the last
 * element of the vector represents the fastest varying
 * dimension and the zeroth element represents the slowest
 * varying, etc.
 *
 * @param[in] size Size of the array.
 *
 */
SNPE_API
void Snpe_TensorShape_Concatenate(Snpe_TensorShape_Handle_t tensorShape, const size_t *dims, size_t size);

/**
 * @brief .
 *
 * @param[in] handle : handle to tensorShape
 *
 * Retrieves the rank i.e. number of dimensions.
 *
 * @return The rank
 */
SNPE_API
size_t Snpe_TensorShape_Rank(Snpe_TensorShape_Handle_t tensorShape);

/**
 * @brief .
 *
 * @param[in] handle : handle to tensorShape
 *
 * @param[in] index : Position in the dimension array.
 *
 * @return The dimension value in tensor shape
 */
SNPE_API
size_t Snpe_TensorShape_At(Snpe_TensorShape_Handle_t tensorShapeHandle, size_t index);

/**
 * @brief Set a value in a TensorShape at the provided index
 *
 * @param[in] handle : handle to tensorShape
 *
 * @param[in] index : Position in the dimension array.
 *
 * @param[in] value : Dimension value to set
 *
 * @return SNPE_SUCCESS on success
 */
SNPE_API
Snpe_ErrorCode_t Snpe_TensorShape_Set(Snpe_TensorShape_Handle_t tensorShapeHandle, size_t index, size_t value);

/**
 * @brief .
 *
 * Retrieves a pointer to the first dimension of shape
 *
 * @param[in] handle : handle to tensorShape
 *
 * @return nullptr if no dimension exists; otherwise, points to
 * the first dimension.
 *
 */
SNPE_API
const size_t* Snpe_TensorShape_GetDimensions(Snpe_TensorShape_Handle_t tensorShape);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_SYSTEM_TENSOR_SHAPE_H
