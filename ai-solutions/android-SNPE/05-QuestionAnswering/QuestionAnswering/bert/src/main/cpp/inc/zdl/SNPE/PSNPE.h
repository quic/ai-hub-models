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

/**
 *  @file
 */

#ifndef _SNPE_PSNPE_H_
#define _SNPE_PSNPE_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlContainer/DlContainer.h"
#include "SNPE/ApplicationBufferMap.h"
#include "SNPE/RuntimeConfigList.h"
#include "SNPE/UserBufferList.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/IBufferAttributes.h"

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"

#include "DlSystem/UserMemoryMap.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A typedef to indicate the callback PSNPE handle of Async Output mode
 */
typedef void* Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t;

//SNPE_API
//Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t Snpe_PSNPE_OutputAsyncCallbackParam_Create(size_t index,
//                                                                                        int status,
//                                                                                        const char* errorMsg);
//
//SNPE_API
//Snpe_ErrorCode_t Snpe_PSNPE_OutputAsyncCallbackParam_Delete(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

// NOTE: we don't need _{Create,Delete} functions because the user does not create or delete these handles
// They're passed in to the callback functions they created

/**
 * @brief Get the data index of an output async PSNPE object
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of output async mode
 *
 * @return The data idx for output async mode
 */
SNPE_API
size_t Snpe_PSNPE_OutputAsyncCallbackParam_GetDataIdx(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

/**
 * @brief Execute an output async PSNPE  object
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of output async mode
 *
 * @return True if executed successfully with outputAsync mode
 */
SNPE_API
int Snpe_PSNPE_OutputAsyncCallbackParam_GetExecuteStatus(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

/**
 * @brief Get the error message during the execution of PSNPE output async mode
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of output async mode
 *
 * @return Error message
 */
SNPE_API
const char* Snpe_PSNPE_OutputAsyncCallbackParam_GetErrorMsg(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

/**
 * @brief Get the ID of an output async PSNPE object
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of output async mode
 *
 * @return The id of an PSNPE object for output async mode
 */
SNPE_API
size_t Snpe_PSNPE_OutputAsyncCallbackParam_GetID(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);



/**
 * A typedef to indicate the output callback of PSNPE handle of input-output async mode
 */
typedef void* Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t;

/**
 * @brief Get the data index of an input-output async PSNPE object
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return The data index for input-output async mode
 */
SNPE_API
size_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetDataIdx(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * @brief Execute an input-output async PSNPE  object
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return True if executed successfully with input-output async mode
 */
SNPE_API
int Snpe_PSNPE_InputOutputAsyncCallbackParam_GetExecuteStatus(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * @brief Get the error message during the execution of PSNPE input-output async mode
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return error message
 */
SNPE_API
const char* Snpe_PSNPE_InputOutputAsyncCallbackParam_GetErrorMsg(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * @brief Get the names of output buffers to the network
 *
 * @param[in] ioacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return Handle of output buffer name list
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetUserBufferNames(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * @brief Get the output buffer map of PSNPE object for input-output async mode
 *
 * @param[in] ioacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return The reference handle of output ApplicationBufferMap
 */
SNPE_API
Snpe_ApplicationBufferMap_Handle_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetOutputMap_Ref(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * @brief Get the id of the output callback for input-output async mode
 *
 * @param[in] oacpHandle Handle to access the PSNPE object of input-output async mode
 *
 * @return The id for output callback for input-output async mode
 */
SNPE_API
size_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetID(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

/**
 * A typedef to indicate the input callback of PSNPE handle of input-output async mode
 */
typedef void* Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t;

/**
 * @brief Get the input list for input callback of input-output async mode
 *
 * @param[in] ioacpHandle Handle to access the object of input callback of input-output async mode
 *
 * @return List the inputs
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputs(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);

/**
 * @brief Get the input names for input callback of input-output async mode
 *
 * @param[in] ioacpHandle Handle to access the object of input callback of input-output async mode
 *
 * @return List the names of input
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputNames(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);

/**
 * @brief Get the id of the input callback for input-output async mode
 *
 * @param[in] oacpHandle Handle to access the object of input-output async mode
 *
 * @return The id of input callback for input-output async mode
 */
SNPE_API
size_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetID(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);

/**
 * @brief A struct to indicate userbuffer data type in output callback of input-output async mode
 */
typedef struct{
  /// data for the one output
  const uint8_t* data;
  /// the data size of this output
  size_t size;
} Snpe_UserBufferData_t;

/**
 * @brief Get the output data of the output callback for input-output async mode
 *
 * @param[in] oacpHandle Handle to access the object of output callback of input-output async mode
 *
 * @param[in] name The output name of output callback of input-output async mode
 *
 * @return The output data of output callback for input-output async mode
 */
SNPE_API
Snpe_UserBufferData_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetUserBuffer(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle,
                                                                             const char* name);
/**
 * A typedef to indicate build configuration
 */
typedef void* Snpe_BuildConfig_Handle_t;

/**
 * A typedef to indicate a PSNPE object
 */
typedef void* Snpe_PSNPE_Handle_t;

/**
 * A typedef to indicate if PSNPE object is built in serial or parallel, default = 0
 */
typedef enum SNPE_API {
  SNPE_PSNPE_BUILDMODE_SERIAL = 0,
  SNPE_PSNPE_BUILDMODE_PARALLEL = 1
} Snpe_PSNPE_BuildMode_t;

/**
 * A typedef to indicate if PSNPE objects are executed in sync mode or output async mode or input-output async mode, default = 0
 */
typedef enum SNPE_API {
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_SYNC = 0,
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_OUTPUTASYNC = 1,
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_INPUTOUTPUTASYNC = 2
} Snpe_PSNPE_InputOutputTransmissionMode_t;

// BuildConfig
/**
 * @brief Create the object of snpe build config
 *
 * @return the SNPE build handle
 */
SNPE_API
Snpe_BuildConfig_Handle_t Snpe_BuildConfig_Create();

/**
 * @brief Release the object of snpe build config
 *
 * @param[in] buildConfigHandle Handle to access the object of snpe buid config
 *
 * @return The error of build config result
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_Delete(Snpe_BuildConfig_Handle_t buildConfigHandle);

/**
 * @brief Get the mode of build snpe object, serial or parallel
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The value of Snpe_PSNPE_BuildMode_t
 */
SNPE_API
Snpe_PSNPE_BuildMode_t Snpe_BuildConfig_GetBuildMode(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set the mode of build snpe object, serial or parallel
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] buildMode A typedef of Snpe_PSNPE_BuildMode_t
 *
 * @return The result of setting mode
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetBuildMode(Snpe_BuildConfig_Handle_t bcHandle, Snpe_PSNPE_BuildMode_t buildMode);

/**
 * @brief Set the dlc model
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] dlcHandle A handle of snpe DLC container
 *
 * @return The result of setting dlc model
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetContainer(Snpe_BuildConfig_Handle_t bcHandle, Snpe_DlContainer_Handle_t dlcHandle);

/**
 * @brief Get dlc container in snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The reference handle of DLC container
 */
SNPE_API
Snpe_DlContainer_Handle_t Snpe_BuildConfig_GetContainer_Ref(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set output buffer names in snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] slHandle A handle of the output layer name list
 *
 * @return The result of setting output names
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputBufferNames(Snpe_BuildConfig_Handle_t bcHandle, Snpe_StringList_Handle_t slHandle);

/**
 * @brief Get output buffer names in snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The reference handle of output buffer name list.
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_BuildConfig_GetOutputBufferNames_Ref(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set output buffer names in snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] slHandle List of tensor names to output. An empty list will result in producing output for the final output tensor of the model. The list will be copied
 *
 * @return The result of setting output tensors
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputTensors(Snpe_BuildConfig_Handle_t bcHandle, Snpe_StringList_Handle_t slHandle);

/**
 * @brief Get output tensors in snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The reference handle of output tensor list
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_BuildConfig_GetOutputTensors_Ref(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set runtime config list for snpe buildConfig
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] rclHandle Handle to access the object of runtime config list
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetRuntimeConfigList(Snpe_BuildConfig_Handle_t bcHandle, Snpe_RuntimeConfigList_Handle_t rclHandle);

/**
 * @brief Get runtime config list for snpe buildConfig
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The reference handle of runtime config list
 */
SNPE_API
Snpe_RuntimeConfigList_Handle_t Snpe_BuildConfig_GetRuntimeConfigList_Ref(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Get input thread number of input data for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The number of input thread
 */
SNPE_API
size_t Snpe_BuildConfig_GetInputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set input thread number of input data for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] threadNumbers The number of input thread for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle, size_t threadNumbers);

/**
 * @brief Get output thread number of output data for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The number of output thread
 */
SNPE_API
size_t Snpe_BuildConfig_GetOutputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set output thread number of output data for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] threadNumbers The number of output thread for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle, size_t threadNumbers);

/**
 * @brief Set output callback for output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] callbackFunc The ouutput callback function for output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                    void (*callbackFunc)(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t));
/**
 * @brief Set the id of output callback function for output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] id The id of output callback function
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);

/**
 * @brief Set the inside output callback handle  to NULL for output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearOutputCallback(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set output callback for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] callbackFunc The output callback function for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                    void (*callbackFunc)(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t));

/**
 * @brief Set the id of output callback function for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] id The id of output callback function for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);

/**
 * @brief Set the inside output callback handle to NULL for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearInputOutputCallback(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set input callback for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] callbackFunc The input callback function for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputInputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                              Snpe_ApplicationBufferMap_Handle_t (*callbackFunc)(
                                                              Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t
                                                              )
                                                              );

/**
 * @brief Set the id of input callback function for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] id The id of input callback function for input-output async mode
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputInputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);

/**
 * @brief Set the inside input callback handle to NULL for input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearInputOutputInputCallback(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set the input and output transmission mode including sync mode, output async mode and input-output async mode, defult is sync mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] iotMode The typedef of Snpe_PSNPE_InputOutputTransmissionMode_t
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputTransmissionMode(Snpe_BuildConfig_Handle_t bcHandle,
                                                                 Snpe_PSNPE_InputOutputTransmissionMode_t iotMode);

/**
 * @brief Get the input and output transmission mode including sync mode, output async mode and input-output async mode
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The typedef of Snpe_PSNPE_InputOutputTransmissionMode_t
 */
SNPE_API
Snpe_PSNPE_InputOutputTransmissionMode_t Snpe_BuildConfig_GetInputOutputTransmissionMode(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set the profiling level for PSNPE build config, default is SNPE_PROFILING_LEVEL_OFF
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] profilingLevel The typedef of Snpe_ProfilingLevel_t
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetProfilingLevel(Snpe_BuildConfig_Handle_t bcHandle, Snpe_ProfilingLevel_t profilingLevel);

/**
 * @brief Get the profiling level for PSNPE build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The typedef of Snpe_ProfilingLevel_t
 */
SNPE_API
Snpe_ProfilingLevel_t Snpe_BuildConfig_GetProfilingLevel(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief To be deprecated, set the encode value when you want to divide one image into 2 or 4 parts to run, default is 0 which means the input don't need dividing.
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] encode0  The uint64 value of encode0
 *
 * @param[in] encode1 The uint64 value of encode1
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode0, uint64_t encode1);

/**
 * @brief To be deprecated, set the encode0 value for snpe build config which is a special feature used in SM8250
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] encode0  The uint64 value of encode0
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode0(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode0);

/**
 * @brief To be deprecated, set the encode1 value for snpe build config which is a special feature used in SM8250
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] encode1  The uint64 value of encode1
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode1(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode1);

/**
 * @brief To be deprecated, get the encode0 and encode1 value for snpe build config which is a special feature used in SM8250
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The uint64 value of encode
 */
SNPE_API
uint64_t* Snpe_BuildConfig_GetEncode(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief To be deprecated, get the encode0 value for snpe build config which is a special feature used in SM8250
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The uint64 value of encode0
 */
SNPE_API
uint64_t Snpe_BuildConfig_GetEncode0(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief To be deprecated, get the encode1 value for snpe build config which is a special feature used in SM8250
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The uint64 value of encode1
 */
SNPE_API
uint64_t Snpe_BuildConfig_GetEncode1(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set true or false for enabling init cache for snpe build config, enabling init cache = 1
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] enableInitCache  True for enabing init cache
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEnableInitCache(Snpe_BuildConfig_Handle_t bcHandle, int enableInitCache);

/**
 * @brief Get the satus of enabling init cache for snpe build config, enabling init cache = 1.
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] enableInitCache  True for enabing init cache
 *
 * @return 1 or 0 for enabling init cache
 */
SNPE_API
int Snpe_BuildConfig_GetEnableInitCache(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Handle needed to access the platformConfig.
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] platformOptions  Options as a const char*
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetPlatformOptions(Snpe_BuildConfig_Handle_t bcHandle, const char* platformOptions);

/**
 * @brief Get the optional platform features for snpe build config
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return Options as a const char*
 */
SNPE_API
const char* Snpe_BuildConfig_GetPlatformOptions(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Set the path directory of output diag log you want to save
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @param[in] diaglogOutputDir The string directory
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetDiaglogOutputDir(Snpe_BuildConfig_Handle_t bcHandle, const char* diaglogOutputDir);

/**
 * @brief Get the path of output diag log
 *
 * @param[in] bcHandle Handle to access the object of snpe buid config
 *
 * @return The string directory
 */
SNPE_API
const char* Snpe_BuildConfig_GetDiaglogOutputDir(Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Create the handle of PSNPE object
 *
 * @return The handle of PSNPE object
 */
SNPE_API
Snpe_PSNPE_Handle_t Snpe_PSNPE_Create();

/**
 * @brief Release the handle of PSNPE object
 *
 * @param[in] psnpeHandle Handle to access the PSNPE object
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Delete(Snpe_PSNPE_Handle_t psnpeHandle);

/**
 * @brief Build the instance of PSNPE object accorading of snpe build config
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Build(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_BuildConfig_Handle_t bcHandle);

/**
 * @brief Execute PSNPE object for sync mode.
 *
 * @param[in] psnpeHandle Handle to access the PSNPE object
 *
 * @param[in] inputBufferListHandle Handle to access the input user buffer list
 *
 * @param[in] outputBufferListHandle Handle to access the output user buffer list
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Execute(Snpe_PSNPE_Handle_t psnpeHandle,
                                    Snpe_UserBufferList_Handle_t inputBufferListHandle,
                                    Snpe_UserBufferList_Handle_t outputBufferListHandle);

/**
 * @brief Execute PSNPE object for input-output async mode
 *
 * @param[in] psnpeHandle Handle to access the PSNPE object
 *
 * @param[in] inputMapHandle Handle to access the input buffer map
 *
 * @param[in] dataIndex The index of input data
 *
 * @param[in] isTF8buff If the input buffer is TF8
 *
 * @param[in] isTF8Outputbuff If the output buffer is TF8
 *
 * @return The result error message
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_ExecuteInputOutputAsync(Snpe_PSNPE_Handle_t psnpeHandle,
                                                    Snpe_StringList_Handle_t inputMapHandle,
                                                    size_t dataIndex,
                                                    int isTF8buff,
                                                    int isTF8Outputbuff);

/**
 * @brief Get the input tensor names for PSNPE object.
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @return The string list of input tensor names
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_GetInputTensorNames(Snpe_PSNPE_Handle_t psnpeHandle);

/**
 * @brief Get the output tensor names for PSNPE object
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @return The string list of output tensor names
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_GetOutputTensorNames(Snpe_PSNPE_Handle_t psnpeHandle);

/**
 * @brief Get the input dimension shape for PSNPE object
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @return The tensor shape of input dimension
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetInputDimensions(Snpe_PSNPE_Handle_t psnpeHandle);

/**
 * @brief Get the input dimension shape for the specific input name for PSNPE object
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @param[in] name The name of input data
 *
 * @return The tensor shape of a specific input name
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetInputDimensions_Name(Snpe_PSNPE_Handle_t psnpeHandle, const char* name);

/**
 * @brief Get the number of elements in each dimension for input and output buffer
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @param[in] name The name of input and output buffer
 *
 * @return Dimension size
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetBufferAttributesDims(Snpe_PSNPE_Handle_t psnpeHandle, const char* name);

/* To be deprecated, please use new api Snpe_PSNPE_RegisterUserMemoryMappedBuffers */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_RegisterIonBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_UserMemoryMap_Handle_t ionBufferMapHandle);

/* To be deprecated, please use new api Snpe_PSNPE_DeregisterUserMemoryMappedBuffers */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_DeregisterIonBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_StringList_Handle_t ionBufferNames);

/**
 * @brief Register Client Memory-Mapped Buffers (Example ION buffers in Android)
 *
 * @param[in] snpeHandle Handle to access the SNPE object
 *
 * @param[in] bufferMapHandle A UserMemoryMap of virtual addresses
 *
 * @note UserBuffer type passed for registration must match the data type of the tensor in the dlc
 *       For regular UserBuffers SNPE performs an online data conversion (quantization or
 *       dequantization etc). This is not possible for memory mapped buffers hence can lead to
 *       issues during execution or accuracy degradation
 *
 * @return SNPE_SUCCESS upon successful memory mapped buffer registration
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_RegisterUserMemoryMappedBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_UserMemoryMap_Handle_t bufferMapHandle);

/**
 * @brief Deregister Client Memory-Mapped Buffers (Example ION buffers in Android)
 *
 * @param[in] snpeHandle Handle to access the SNPE object
 *
 * @param[in] bufferNamesHandle A StringList of memory mapped buffer names
 *
 * @return SNPE_SUCCESS upon successful memory mapped buffer deregistration
 */
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_DeregisterUserMemoryMappedBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_StringList_Handle_t bufferNamesHandle);

/**
 * @brief Get the error message during the failed execution
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @return The error message
 */
SNPE_API
const char* Snpe_PSNPE_GetLastErrorString(Snpe_PSNPE_Handle_t psnpeHandle);

/**
 * @brief Get the handle of IBufferAttributes
 *
 * @param[in] bcHandle Handle to access the PSNPE object
 *
 * @param[in] name The name of attribute buffer
 *
 * @return Handle to access IBufferAttributes
 */
SNPE_API
Snpe_IBufferAttributes_Handle_t Snpe_PSNPE_GetInputOutputBufferAttributes(Snpe_PSNPE_Handle_t psnpeHandle, const char *name);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_PSNPE_H_
