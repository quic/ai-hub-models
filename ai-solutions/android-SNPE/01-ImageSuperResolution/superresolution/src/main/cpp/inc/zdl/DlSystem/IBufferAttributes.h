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
// Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 */

#ifndef _IBUFFER_ATTRIBUTES_H
#define _IBUFFER_ATTRIBUTES_H

#include "DlSystem/IUserBuffer.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/DlError.h"
#include "DlSystem/SnpeApiExportDefine.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE IBufferAttributes handle
 */
typedef void* Snpe_IBufferAttributes_Handle_t;


/**
 * @brief Gets the buffer's element size, in bytes
 *
 * This can be used to compute the memory size required
 * to back this buffer.
 *
 * @param[in] handle : Handle to access IBufferAttributes
 *
 * @return Element size, in bytes
 */
SNPE_API
size_t Snpe_IBufferAttributes_GetElementSize(Snpe_IBufferAttributes_Handle_t handle);

/**
 * @brief Gets the element's encoding type
 *
 * @param[in] handle : Handle to access IBufferAttributes
 *
 * @return encoding type
 */
SNPE_API
Snpe_UserBufferEncoding_ElementType_t Snpe_IBufferAttributes_GetEncodingType(Snpe_IBufferAttributes_Handle_t handle);

/**
 * @brief Gets the number of elements in each dimension
 *
 * @param[in] handle : Handle to access IBufferAttributes
 *
 * @return Dimension size, in terms of number of elements
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_IBufferAttributes_GetDims(Snpe_IBufferAttributes_Handle_t handle);

/**
 * @brief Gets the alignment requirement of each dimension
 *
 * Alignment per each dimension is expressed as an multiple, for
 * example, if one particular dimension can accept multiples of 8,
 * the alignment will be 8.
 *
 * @param[in] handle : Handle to access IBufferAttributes
 *
 * @return Alignment in each dimension, in terms of multiple of
 *         number of elements
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_IBufferAttributes_GetAlignments(Snpe_IBufferAttributes_Handle_t handle);

/**
 * @brief Gets the buffer encoding returned from the network responsible
 * for generating this buffer. Depending on the encoding type, this will
 * be an instance of an encoding type specific derived class.
 *
 * @param[in] handle : Handle to access IBufferAttributes
 *
 * @return Derived user buffer encoding object.
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_IBufferAttributes_GetEncoding_Ref(Snpe_IBufferAttributes_Handle_t handle);

/**
 * @brief Destroys the IBufferAttributes object
 *
 * @param[handle] handle : Handle to access IBufferAttributes
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IBufferAttributes_Delete(Snpe_IBufferAttributes_Handle_t handle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _IBUFFER_ATTRIBUTES_H
