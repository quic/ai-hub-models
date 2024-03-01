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

#ifndef _DL_SYSTEM_ITENSOR_H_
#define _DL_SYSTEM_ITENSOR_H_

#include <stdint.h>

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/DlError.h"


#ifdef __cplusplus
extern "C" {
#endif


/**
 * Represents a tensor which holds n-dimensional data. It is important to
 * understand how the tensor data is represented in memory
 * relative to the tensor dimensions. Tensors store data in
 * memory in row-major order (i.e. the last tensor dimension is
 * the fastest varying one). For example, if you have a two
 * dimensional tensor with 3 rows and 2 columns (i.e. the tensor
 * dimensions are 3,2 as returned in tensor dimension vectors)
 * with the following data in terms rows and columns:
 *
 * | 1 2 | <br/>
 * | 3 4 | <br/>
 * | 5 6 | <br/>
 *
 * This data would be stored in memory as 1,2,3,4,5,6.
 */
typedef void* Snpe_ITensor_Handle_t;


/**
 * Destroys/frees an ITensor
 *
 * @param[in] userBufferHandle : Handle to access the IUserBuffer
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_ITensor_Delete(Snpe_ITensor_Handle_t iTensorHandle);

/**
 * Returns a tensor iterator pointing to the beginning
 * of the data in the tensor.
 *
 * @param[in] tensorHandle : Handle to access ITensor
 *
 * @return The tensor data as a void pointer.
 */
SNPE_API
void* Snpe_ITensor_GetData(Snpe_ITensor_Handle_t tensorHandle);

/**
 * @brief Gets the shape of this tensor.
 *
 * The last element of the vector represents the fastest varying
 * dimension and the zeroth element represents the slowest
 * varying dimension, etc.
 *
 * @param[in] tensorHandle : Handle to access ITensor
 *
 * @return A TensorShape handle holding the tensor dimensions.
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_ITensor_GetShape(Snpe_ITensor_Handle_t tensorHandle);

/**
 * Returns the element size of the data in the tensor
 * (discounting strides). This is how big a buffer would
 * need to be to hold the tensor data contiguously in
 * memory.
 *
 * @param[in] tensorHandle : Handle to access ITensor
 *
 * @return The size of the tensor (in elements).
 */
SNPE_API
size_t Snpe_ITensor_GetSize(Snpe_ITensor_Handle_t tensorHandle);

SNPE_API
int Snpe_ITensor_IsQuantized(Snpe_ITensor_Handle_t tensorHandle);

SNPE_API
float Snpe_ITensor_GetDelta(Snpe_ITensor_Handle_t tensorHandle);

SNPE_API
float Snpe_ITensor_GetOffset(Snpe_ITensor_Handle_t tensorHandle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _DL_SYSTEM_ITENSOR_H_
