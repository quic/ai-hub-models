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
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _SNPE_APPLICATION_BUFFER_MAP_H_
#define _SNPE_APPLICATION_BUFFER_MAP_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif


#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"
#include "DlSystem/StringList.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef void* Snpe_ApplicationBufferMap_Handle_t;

SNPE_API
Snpe_ApplicationBufferMap_Handle_t Snpe_ApplicationBufferMap_Create();

SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_Delete(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_Add(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle,
                                               const char* name,
                                               const uint8_t* buff,
                                               size_t size);

SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_AddFloat(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle,
                                                    const char* name,
                                                    const float* buff,
                                                    size_t size);

SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_Remove(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle,
                                                  const char* name);

SNPE_API
size_t Snpe_ApplicationBufferMap_Size(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_Clear(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle);


SNPE_API
Snpe_StringList_Handle_t Snpe_ApplicationBufferMap_GetUserBufferNames(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_ApplicationBufferMap_GetUserBuffer(Snpe_ApplicationBufferMap_Handle_t applicationBufferMapHandle,
                                                         const char* name,
                                                         size_t* size,
                                                         const uint8_t** data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_APPLICATION_BUFFER_MAP_H_
