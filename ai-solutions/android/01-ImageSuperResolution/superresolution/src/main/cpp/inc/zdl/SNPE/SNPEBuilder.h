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

#ifndef _SNPE_BUILDER_H_
#define _SNPE_BUILDER_H_

#include "SNPE/SNPE.h"
#include "DlSystem/DlEnums.h"
#include "DlSystem/DlError.h"
#include "DlSystem/IOBufferDataTypeMap.h"
#include "DlSystem/TensorShapeMap.h"
#include "DlSystem/RuntimeList.h"
#include "DlSystem/PlatformConfig.h"
#include "DlContainer/DlContainer.h"

#ifdef __cplusplus
extern "C" {
#endif



/**
 * A typedef to indicate a SNPEBuilder handle
 */
typedef void* Snpe_SNPEBuilder_Handle_t;

/**
 * The builder class for creating SNPE objects.
 * Not meant to be extended.
 */


/**
 * @brief Constructor of NeuralNetwork Builder ith a supplied model.
 *
 * @param[in] containerHandle A DlContainer holding the model.
 *
 * @return A new instance of a SNPEBuilder object
 *         that can be used to configure and build
 *         an instance of SNPE.
 *
 */
SNPE_API
Snpe_SNPEBuilder_Handle_t Snpe_SNPEBuilder_Create(Snpe_DlContainer_Handle_t containerHandle);

/**
 * Destroys/frees a SNPEBuilder object
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_Delete(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle);

/**
 * @brief Requests a performance profile.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] performanceProfile The target performance profile.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetPerformanceProfile(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_PerformanceProfile_t performanceProfile);

/**
 * @brief Sets the profiling level. Default profiling level for
 *        SNPEBuilder is off. Off and basic only applies to DSP runtime.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] profilingLevel The target profiling level.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetProfilingLevel(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_ProfilingLevel_t profilingLevel);

/**
 * @brief Sets a preference for execution priority.
 *
 * This allows the caller to give coarse hint to SNPE runtime
 * about the priority of the network.  SNPE runtime is free to use
 * this information to co-ordinate between different workloads
 * that may or may not extend beyond SNPE.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] priority The target performance profile.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetExecutionPriorityHint(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_ExecutionPriorityHint_t priority);

/**
 * @brief Sets the layers that will generate output.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] outputLayerNames List of layer names to
 *                             output. An empty list will
 *                             result in only the final
 *                             layer of the model being
 *                             the output layer.  The list
 *                             will be copied.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetOutputLayers(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_StringList_Handle_t outputLayerNames);

/**
 * @brief Sets the output tensor names.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] outputTensorNames List of tensor names to
 *                             output. An empty list will
 *                             result in producing output for the final
 *                             output tensor of the model.
 *                             The list will be copied.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetOutputTensors(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_StringList_Handle_t outputTensorNames);

/**
 * @brief Sets whether this neural network will perform inference with
 *        input from user-supplied buffers, and write output to user-supplied
 *        buffers.  Default behaviour is to use tensors created by
 *        ITensorFactory.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] bufferMode Boolean whether to use user-supplied buffer or not.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, int bufferMode);

/**
 * @brief Sets the debug mode of the runtime.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] debugMode This enables debug mode for the runtime. It
 *                      does two things. For an empty
 *                      outputLayerNames list, all layers will be
 *                      output. It might also disable some internal
 *                      runtime optimizations (e.g., some networks
 *                      might be optimized by combining layers,
 *                      etc.).
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetDebugMode(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, int debugMode);



/**
 * @brief Sets network's input dimensions to enable resizing of
 *        the spatial dimensions of each layer for fully convolutional networks,
 *        and the batch dimension for all networks.
 *
 * @param[in] tensorShapeMapHandle : Handle to the map of input names and their new dimensions.
 *                           The new dimensions overwrite the input dimensions
 *                           embedded in the model and then resize each layer
 *                           of the model. If the model contains
 *                           layers whose dimensions cannot be resized e.g FullyConnected,
 *                           exception will be thrown when SNPE instance is actually built.
 *                           In general the batch dimension is always resizable.
 *                           After resizing of layers' dimensions in model based
 *                           on new input dimensions, the new model is revalidated
 *                           against all runtime constraints, whose failures may
 *                           result in cpu fallback situation.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetInputDimensions(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_TensorShapeMap_Handle_t inputDimensionsMapHandle);

/**
 * @brief Sets the mode of init caching functionality.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] mode   Boolean. This flag enables/disables the functionality of init caching.
 *                   When init caching functionality is enabled, a set of init caches
 *                   will be created during network building/initialization process
 *                   and will be added to DLC container. If such DLC container is saved
 *                   by the user, in subsequent network building/initialization processes
 *                   these init caches will be loaded from the DLC so as to reduce initialization time.
 *                   In disable mode, no init caches will be added to DLC container.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetInitCacheMode(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, int cacheMode);

/**
 * @brief Returns an instance of SNPE based on the current parameters.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @return A new instance of a @ref CAPI_SNPE "SNPE" object that can be used
 *         to execute models or null if any errors occur.
 */
SNPE_API
Snpe_SNPE_Handle_t Snpe_SNPEBuilder_Build(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle);

/**
 * @brief Sets the platform configuration.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] platformConfig The platform configuration.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetPlatformConfig(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_PlatformConfig_Handle_t platformConfigHandle);

/**
 * @brief Sets network's runtime order of precedence. Example:
 *        CPU_FLOAT32, GPU_FLOAT16, AIP_FIXED8_TF
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] runtimeListHandle The list of runtime in order of precedence
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetRuntimeProcessorOrder(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_RuntimeList_Handle_t runtimeListHandle);

/**
 * @brief Sets the unconsumed tensors as output
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] setOutput Boolean. This enables unconsumed tensors (i.e)
 *                      outputs which are not inputs to any
 *                      layer (basically dead ends) to be marked
 *                      for output
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetUnconsumedTensorsAsOutputs(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, int setOutput);

/**
 * @brief Execution terminated when exceeding time limit.
 *        Only valid for dsp runtime currently.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] timeout Time limit value in microseconds
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetTimeOut(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, uint64_t timeoutMicroSec);


/**
 * @brief Sets the datatype of the buffer.
 *        Only valid for dsp runtime currently.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] dataTypeMapHandle Map of the buffer names and the datatype that needs to be set.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetBufferDataType(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, Snpe_IOBufferDataTypeMap_Handle_t dataTypeMapHandle);

/**
 * @brief Sets up the entire initialization callflow to
 *        happen on the user's thread
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] singleThreadedInit Flag indicating user's intent to perform initialization callflow
 *            on caller's thread.
 *            When set to 1, initialization will happen on the user's thread
 *            When set to 0, initialization will happen on a new thread. This is the default
 *            behavior (analogous to not calling this API)
*/
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetSingleThreadedInit(Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, int singleThreadedInit);

/**
 * @brief Sets the fixed point execution mode for CPU runtime.
 *        If a floating point DLC is executed with this option set, the program will be terminated with an exception.
 *        If a quantized DLC is executed without this option set, the execution will be in floating point mode in CPU.
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] cpuFxpMode Boolean If set to true, enables the fixed point mode.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetCpuFixedPointMode(
      Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, bool cpuFxpMode);

/**
 * @brief Sets model name for logging
 *
 * @param[in] snpeBuilderHandle Handle to access the SNPEBuilder object
 *
 * @param[in] modelName String Model name for logging.
 *
 */
SNPE_API
Snpe_ErrorCode_t Snpe_SNPEBuilder_SetModelName(
      Snpe_SNPEBuilder_Handle_t snpeBuilderHandle, const char *modelName);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_BUILDER_H_
