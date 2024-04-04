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
//  Copyright (c) 2022,2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _SNPE_USER_BUFFER_LIST_H_
#define _SNPE_USER_BUFFER_LIST_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"

#include "DlSystem/UserBufferMap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* Snpe_UserBufferList_Handle_t;

SNPE_API
Snpe_UserBufferList_Handle_t Snpe_UserBufferList_Create();

SNPE_API
Snpe_UserBufferList_Handle_t Snpe_UserBufferList_CreateCopy(Snpe_UserBufferList_Handle_t userBufferListHandle);

SNPE_API
Snpe_UserBufferList_Handle_t Snpe_UserBufferList_CreateSize(size_t size);

SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferList_Delete(Snpe_UserBufferList_Handle_t userBufferListHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferList_PushBack(Snpe_UserBufferList_Handle_t userBufferListHandle,
                                              Snpe_UserBufferMap_Handle_t userBufferMapHandle);

SNPE_API
Snpe_UserBufferMap_Handle_t Snpe_UserBufferList_At_Ref(Snpe_UserBufferList_Handle_t userBufferListHandle,
                                                   size_t idx);

SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferList_Assign(Snpe_UserBufferList_Handle_t srcUserBufferListHandle,
                                            Snpe_UserBufferList_Handle_t dstUserBufferListHandle);

SNPE_API
size_t Snpe_UserBufferList_Size(Snpe_UserBufferList_Handle_t userBufferListHandle);

SNPE_API
size_t Snpe_UserBufferList_Capacity(Snpe_UserBufferList_Handle_t userBufferListHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferList_Clear(Snpe_UserBufferList_Handle_t userBufferListHandle);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_USER_BUFFER_LIST_H_
