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

#ifndef _DL_ENUMS_H_
#define _DL_ENUMS_H_

#include "DlSystem/SnpeApiExportDefine.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Enumeration of supported target runtimes.
 */
typedef enum
{
  /// Special value indicating the property is unset.
  SNPE_RUNTIME_UNSET = -1,
  /// Run the processing on Snapdragon CPU.
  /// Data: float 32bit
  /// Math: float 32bit
  SNPE_RUNTIME_CPU_FLOAT32  = 0,
  /// Default legacy enum to retain backward compatibility.
  /// CPU = CPU_FLOAT32
  SNPE_RUNTIME_CPU = SNPE_RUNTIME_CPU_FLOAT32,

  /// Run the processing on the Adreno GPU.
  /// Data: float 16bit
  /// Math: float 32bit
  SNPE_RUNTIME_GPU_FLOAT32_16_HYBRID = 1,
  /// Default legacy enum to retain backward compatibility.
  /// GPU = GPU_FLOAT32_16_HYBRID
  SNPE_RUNTIME_GPU = SNPE_RUNTIME_GPU_FLOAT32_16_HYBRID,

  /// Run the processing on the Hexagon DSP.
  /// Data: 8bit fixed point Tensorflow style format
  /// Math: 8bit fixed point Tensorflow style format
  SNPE_RUNTIME_DSP_FIXED8_TF = 2,
  /// Default legacy enum to retain backward compatibility.
  /// DSP = DSP_FIXED8_TF
  SNPE_RUNTIME_DSP = SNPE_RUNTIME_DSP_FIXED8_TF,

  /// Run the processing on the Adreno GPU.
  /// Data: float 16bit
  /// Math: float 16bit
  SNPE_RUNTIME_GPU_FLOAT16 = 3,

  /// Run the processing on Snapdragon AIX+HVX.
  /// Data: 8bit fixed point Tensorflow style format
  /// Math: 8bit fixed point Tensorflow style format
  SNPE_RUNTIME_AIP_FIXED8_TF = 5,
  SNPE_RUNTIME_AIP_FIXED_TF = SNPE_RUNTIME_AIP_FIXED8_TF
} Snpe_Runtime_t;

/**
 * Enumeration of runtime available check options.
 */
typedef enum
{
  /// Perform standard runtime available check
  SNPE_RUNTIME_CHECK_OPTION_DEFAULT = 2,
  /// Perform standard runtime available check
  SNPE_RUNTIME_CHECK_OPTION_NORMAL_CHECK = 0,
  /// Perform basic runtime available check, may be runtime specific
  SNPE_RUNTIME_CHECK_OPTION_BASIC_CHECK = 1,
  /// Perform unsignedPD runtime available check
  SNPE_RUNTIME_CHECK_OPTION_UNSIGNEDPD_CHECK = 2,
} Snpe_RuntimeCheckOption_t;

/**
 * Enumeration of various performance profiles that can be requested.
 */
typedef enum
{
  /// Run in a standard mode.
  /// This mode will be deprecated in the future and replaced with BALANCED.
  SNPE_PERFORMANCE_PROFILE_DEFAULT = 0,
  /// Run in a balanced mode.
  SNPE_PERFORMANCE_PROFILE_BALANCED = 0,

  /// Run in high performance mode
  SNPE_PERFORMANCE_PROFILE_HIGH_PERFORMANCE = 1,

  /// Run in a power sensitive mode, at the expense of performance.
  SNPE_PERFORMANCE_PROFILE_POWER_SAVER = 2,

  /// Use system settings.  SNPE makes no calls to any performance related APIs.
  SNPE_PERFORMANCE_PROFILE_SYSTEM_SETTINGS = 3,

  /// Run in sustained high performance mode
  SNPE_PERFORMANCE_PROFILE_SUSTAINED_HIGH_PERFORMANCE = 4,

  /// Run in burst mode
  SNPE_PERFORMANCE_PROFILE_BURST = 5,

  /// Run in lower clock than POWER_SAVER, at the expense of performance.
  SNPE_PERFORMANCE_PROFILE_LOW_POWER_SAVER = 6,

  /// Run in higher clock and provides better performance than POWER_SAVER.
  SNPE_PERFORMANCE_PROFILE_HIGH_POWER_SAVER = 7,

  /// Run in lower balanced mode
  SNPE_PERFORMANCE_PROFILE_LOW_BALANCED = 8,

  /// Run in lowest clock at the expense of performance
  SNPE_PERFORMANCE_PROFILE_EXTREME_POWER_SAVER = 9,

} Snpe_PerformanceProfile_t;

/**
 * Enumeration of various profilngLevels that can be requested.
 */
typedef enum
{
  /// No profiling.
  /// Collects no runtime stats in the DiagLog
  SNPE_PROFILING_LEVEL_OFF = 0,

  /// Basic profiling
  /// Collects some runtime stats in the DiagLog
  SNPE_PROFILING_LEVEL_BASIC = 1,

  /// Detailed profiling
  /// Collects more runtime stats in the DiagLog, including per-layer statistics
  /// Performance may be impacted
  SNPE_PROFILING_LEVEL_DETAILED = 2,

  /// Moderate profiling
  /// Collects more runtime stats in the DiagLog, no per-layer statistics
  SNPE_PROFILING_LEVEL_MODERATE = 3,

  /// Linting profiling
  /// HTP exclusive profiling level that collects in-depth performance metrics
  /// for each op in the graph including main thread execution time and time spent
  /// on parallel background ops
  SNPE_PROFILING_LEVEL_LINTING = 4

} Snpe_ProfilingLevel_t;

/**
 * Enumeration of various execution priority hints.
 */
typedef enum
{
  /// Normal priority
  SNPE_EXECUTION_PRIORITY_NORMAL = 0,

  /// Higher than normal priority
  SNPE_EXECUTION_PRIORITY_HIGH = 1,

  /// Lower priority
  SNPE_EXECUTION_PRIORITY_LOW = 2,

  /// Between Normal and High priority
  SNPE_EXECUTION_PRIORITY_NORMAL_HIGH = 3

} Snpe_ExecutionPriorityHint_t;

/**
 * Enumeration that lists the supported image encoding formats.
 */
typedef enum
{
  /// For unknown image type. Also used as a default value for ImageEncoding_t.
  SNPE_IMAGE_ENCODING_UNKNOWN = 0,

  /// The RGB format consists of 3 bytes per pixel: one byte for
  /// Red, one for Green, and one for Blue. The byte ordering is
  /// endian independent and is always in RGB byte order.
  SNPE_IMAGE_ENCODING_RGB = 1,

  /// The ARGB32 format consists of 4 bytes per pixel: one byte for
  /// Red, one for Green, one for Blue, and one for the alpha channel.
  /// The alpha channel is ignored. The byte ordering depends on the
  /// underlying CPU. For little endian CPUs, the byte order is BGRA.
  /// For big endian CPUs, the byte order is ARGB.
  SNPE_IMAGE_ENCODING_ARGB32 = 2,

  /// The RGBA format consists of 4 bytes per pixel: one byte for
  /// Red, one for Green, one for Blue, and one for the alpha channel.
  /// The alpha channel is ignored. The byte ordering is endian independent
  /// and is always in RGBA byte order.
  SNPE_IMAGE_ENCODING_RGBA = 3,

  /// The GRAYSCALE format is for 8-bit grayscale.
  SNPE_IMAGE_ENCODING_GRAYSCALE = 4,

  /// NV21 is the Android version of YUV. The Chrominance is down
  /// sampled and has a subsampling ratio of 4:2:0. Note that this
  /// image format has 3 channels, but the U and V channels
  /// are subsampled. For every four Y pixels there is one U and one V pixel. @newpage
  SNPE_IMAGE_ENCODING_NV21 = 5,

  /// The BGR format consists of 3 bytes per pixel: one byte for
  /// Red, one for Green and one for Blue. The byte ordering is
  /// endian independent and is always BGR byte order.
  SNPE_IMAGE_ENCODING_BGR = 6
} Snpe_ImageEncoding_t;

/**
 * Enumeration that lists the supported LogLevels that can be set by users.
 */
typedef enum
{
  /// Enumeration variable to be used by user to set logging level to FATAL.
  SNPE_LOG_LEVEL_FATAL = 0,

  /// Enumeration variable to be used by user to set logging level to ERROR.
  SNPE_LOG_LEVEL_ERROR = 1,

  /// Enumeration variable to be used by user to set logging level to WARN.
  SNPE_LOG_LEVEL_WARN = 2,

  /// Enumeration variable to be used by user to set logging level to INFO.
  SNPE_LOG_LEVEL_INFO = 3,

  /// Enumeration variable to be used by user to set logging level to VERBOSE.
  SNPE_LOG_LEVEL_VERBOSE = 4
} Snpe_LogLevel_t;

/**
 * Enumeration that list the supported data types for buffers
 */
typedef enum
{
  /// Unspecified
  SNPE_IO_BUFFER_DATATYPE_UNSPECIFIED = 0,

  /// 32-bit floating point
  SNPE_IO_BUFFER_DATATYPE_FLOATING_POINT_32 = 1,

  /// 16-bit floating point
  SNPE_IO_BUFFER_DATATYPE_FLOATING_POINT_16 = 2,

  /// 8-bit fixed point
  SNPE_IO_BUFFER_DATATYPE_FIXED_POINT_8 =  3,

  /// 16-bit fixed point
  SNPE_IO_BUFFER_DATATYPE_FIXED_POINT_16 = 4
} Snpe_IOBufferDataType_t;


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _DL_ENUMS_H_
