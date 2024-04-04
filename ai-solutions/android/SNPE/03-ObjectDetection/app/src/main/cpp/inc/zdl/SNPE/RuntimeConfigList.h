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

#ifndef _SNPE_RUNTIME_CONFIG_LIST_H_
#define _SNPE_RUNTIME_CONFIG_LIST_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"

#include "DlSystem/DlEnums.h"
#include "DlSystem/RuntimeList.h"
#include "DlSystem/TensorShapeMap.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef void* Snpe_RuntimeConfig_Handle_t;

SNPE_API
Snpe_RuntimeConfig_Handle_t Snpe_RuntimeConfig_Create();

SNPE_API
Snpe_RuntimeConfig_Handle_t Snpe_RuntimeConfig_CreateCopy(Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_Delete(Snpe_RuntimeConfig_Handle_t rcHandle);


SNPE_API
Snpe_Runtime_t Snpe_RuntimeConfig_GetRuntime(Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_SetRuntime(Snpe_RuntimeConfig_Handle_t rcHandle, Snpe_Runtime_t runtime);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_SetRuntimeList(Snpe_RuntimeConfig_Handle_t rcHandle, Snpe_RuntimeList_Handle_t rlHandle);

SNPE_API
Snpe_RuntimeList_Handle_t Snpe_RuntimeConfig_GetRuntimeList_Ref(Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_PerformanceProfile_t Snpe_RuntimeConfig_GetPerformanceProfile(Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_SetPerformanceProfile(Snpe_RuntimeConfig_Handle_t rcHandle, Snpe_PerformanceProfile_t perfProfile);

SNPE_API
int Snpe_RuntimeConfig_GetEnableCPUFallback(Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_SetEnableCPUFallback(Snpe_RuntimeConfig_Handle_t rcHandle, int enableCpuFallback);


SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfig_SetInputDimensionsMap(Snpe_RuntimeConfig_Handle_t rcHandle, Snpe_TensorShapeMap_Handle_t tsmHandle);

SNPE_API
Snpe_TensorShapeMap_Handle_t Snpe_RuntimeConfig_GetInputDimensionsMap_Ref(Snpe_RuntimeConfig_Handle_t rcHandle);



typedef void* Snpe_RuntimeConfigList_Handle_t;

SNPE_API
Snpe_RuntimeConfigList_Handle_t Snpe_RuntimeConfigList_Create();

SNPE_API
Snpe_RuntimeConfigList_Handle_t Snpe_RuntimeConfigList_CreateSize(size_t size);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfigList_Delete(Snpe_RuntimeConfigList_Handle_t rclHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfigList_PushBack(Snpe_RuntimeConfigList_Handle_t rclHandle, Snpe_RuntimeConfig_Handle_t rcHandle);

SNPE_API
Snpe_RuntimeConfig_Handle_t Snpe_RuntimeConfigList_At_Ref(Snpe_RuntimeConfigList_Handle_t rclHandle, size_t idx);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfigList_Assign(Snpe_RuntimeConfigList_Handle_t rclSrcHandle, Snpe_RuntimeConfigList_Handle_t rclDstHandle);

SNPE_API
size_t Snpe_RuntimeConfigList_Size(Snpe_RuntimeConfigList_Handle_t rclHandle);

SNPE_API
size_t Snpe_RuntimeConfigList_Capacity(Snpe_RuntimeConfigList_Handle_t rclHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_RuntimeConfigList_Clear(Snpe_RuntimeConfigList_Handle_t rclHandle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_RUNTIME_CONFIG_LIST_H_
