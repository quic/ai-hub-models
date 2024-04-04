//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include "Wrapper.hpp"

namespace DlSystem {
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Enumeration of supported target runtimes.
 */
enum class Runtime_t
{
  /// Special value indicating the property is unset.
  UNSET = -1,
  /// Run the processing on Snapdragon CPU.
  /// Data: float 32bit
  /// Math: float 32bit
  CPU_FLOAT32  = 0,
  /// Default legacy enum to retain backward compatibility.
  /// CPU = CPU_FLOAT32
  CPU = CPU_FLOAT32,

  /// Run the processing on the Adreno GPU.
  /// Data: float 16bit
  /// Math: float 32bit
  GPU_FLOAT32_16_HYBRID = 1,
  /// Default legacy enum to retain backward compatibility.
  /// GPU = GPU_FLOAT32_16_HYBRID
  GPU = GPU_FLOAT32_16_HYBRID,

  /// Run the processing on the Hexagon DSP.
  /// Data: 8bit fixed point Tensorflow style format
  /// Math: 8bit fixed point Tensorflow style format
  DSP_FIXED8_TF = 2,
  /// Default legacy enum to retain backward compatibility.
  /// DSP = DSP_FIXED8_TF
  DSP = DSP_FIXED8_TF,

  /// Run the processing on the Adreno GPU.
  /// Data: float 16bit
  /// Math: float 16bit
  GPU_FLOAT16 = 3,

  /// Run the processing on Snapdragon AIX+HVX.
  /// Data: 8bit fixed point Tensorflow style format
  /// Math: 8bit fixed point Tensorflow style format
  AIP_FIXED8_TF = 5,
  AIP_FIXED_TF = AIP_FIXED8_TF,

  /// Any new enums should be added above this line
  NUM_RUNTIME_TARGETS
};

/**
 * Enumeration of runtime available check options.
 */
enum class RuntimeCheckOption_t
{
  /// Perform standard runtime available check
  NORMAL_CHECK = 0,
  /// Perform basic runtime available check, may be runtime specific
  BASIC_CHECK = 1,
  /// Perform unsignedPD runtime available check
  UNSIGNEDPD_CHECK = 2,
  /// Perform standard runtime available check
  DEFAULT = 2,
  /// Any new enums should be added above this line
  NUM_RUNTIMECHECK_OPTIONS
};

/**
 * Enumeration of various performance profiles that can be requested.
 */
enum class PerformanceProfile_t
{
  /// Run in a standard mode.
  /// This mode will be deprecated in the future and replaced with BALANCED.
  DEFAULT = 0,
  /// Run in a balanced mode.
  BALANCED = 0,

  /// Run in high performance mode
  HIGH_PERFORMANCE = 1,

  /// Run in a power sensitive mode, at the expense of performance.
  POWER_SAVER = 2,

  /// Use system settings.  SNPE makes no calls to any performance related APIs.
  SYSTEM_SETTINGS = 3,

  /// Run in sustained high performance mode
  SUSTAINED_HIGH_PERFORMANCE = 4,

  /// Run in burst mode
  BURST = 5,

  /// Run in lower clock than POWER_SAVER, at the expense of performance.
  LOW_POWER_SAVER = 6,

  /// Run in higher clock and provides better performance than POWER_SAVER.
  HIGH_POWER_SAVER = 7,

  /// Run in lower balanced mode
  LOW_BALANCED = 8,

  /// Run in lowest clock at the expense of performance
  EXTREME_POWER_SAVER = 9,

  /// Any new enums should be added above this line
  NUM_PERF_PROFILES
};

/**
 * Enumeration of various profilngLevels that can be requested.
 */
enum class ProfilingLevel_t
{
  /// No profiling.
  /// Collects no runtime stats in the DiagLog
  OFF = 0,

  /// Basic profiling
  /// Collects some runtime stats in the DiagLog
  BASIC = 1,

  /// Detailed profiling
  /// Collects more runtime stats in the DiagLog, including per-layer statistics
  /// Performance may be impacted
  DETAILED = 2,

  /// Moderate profiling
  /// Collects more runtime stats in the DiagLog, no per-layer statistics
  MODERATE = 3,

  /// Linting profiling
  /// HTP exclusive profiling level that collects in-depth performance metrics
  /// for each op in the graph including main thread execution time and time spent
  /// on parallel background ops
  LINTING = 4
};

/**
 * Enumeration of various execution priority hints.
 */
enum class ExecutionPriorityHint_t
{
  /// Normal priority
  NORMAL = 0,

  /// Higher than normal priority
  HIGH = 1,

  /// Lower priority
  LOW = 2,

  /// Between Normal and High priority
  NORMAL_HIGH = 3,

  /// Any new enums should be added above this line
  NUM_EXECUTION_PRIORITY_HINTS
};

/** @} */ /* end_addtogroup c_plus_plus_apis C++*/

/**
 * Enumeration that lists the supported image encoding formats.
 */
enum class ImageEncoding_t
{
  /// For unknown image type. Also used as a default value for ImageEncoding_t.
  UNKNOWN = 0,

  /// The RGB format consists of 3 bytes per pixel: one byte for
  /// Red, one for Green, and one for Blue. The byte ordering is
  /// endian independent and is always in RGB byte order.
  RGB = 1,

  /// The ARGB32 format consists of 4 bytes per pixel: one byte for
  /// Red, one for Green, one for Blue, and one for the alpha channel.
  /// The alpha channel is ignored. The byte ordering depends on the
  /// underlying CPU. For little endian CPUs, the byte order is BGRA.
  /// For big endian CPUs, the byte order is ARGB.
  ARGB32 = 2,

  /// The RGBA format consists of 4 bytes per pixel: one byte for
  /// Red, one for Green, one for Blue, and one for the alpha channel.
  /// The alpha channel is ignored. The byte ordering is endian independent
  /// and is always in RGBA byte order.
  RGBA = 3,

  /// The GRAYSCALE format is for 8-bit grayscale.
  GRAYSCALE = 4,

  /// NV21 is the Android version of YUV. The Chrominance is down
  /// sampled and has a subsampling ratio of 4:2:0. Note that this
  /// image format has 3 channels, but the U and V channels
  /// are subsampled. For every four Y pixels there is one U and one V pixel. @newpage
  NV21 = 5,

  /// The BGR format consists of 3 bytes per pixel: one byte for
  /// Red, one for Green and one for Blue. The byte ordering is
  /// endian independent and is always BGR byte order.
  BGR = 6
};

/**
 * Enumeration that lists the supported LogLevels that can be set by users.
 */
enum class LogLevel_t
{
  /// Enumeration variable to be used by user to set logging level to FATAL.
  LOG_FATAL = 0,

  /// Enumeration variable to be used by user to set logging level to ERROR.
  LOG_ERROR = 1,

  /// Enumeration variable to be used by user to set logging level to WARN.
  LOG_WARN = 2,

  /// Enumeration variable to be used by user to set logging level to INFO.
  LOG_INFO = 3,

  /// Enumeration variable to be used by user to set logging level to VERBOSE.
  LOG_VERBOSE = 4,

  /// Any new enums should be added above this line
  NUM_LOG_LEVELS
};

enum class IOBufferDataType_t : int
{
  UNSPECIFIED = 0,
  FLOATING_POINT_32 = 1,
  FLOATING_POINT_16 = 2,
  FIXED_POINT_8 =  3,
  FIXED_POINT_16 = 4,
  INT_32 = 5,
  UINT_32 = 6,
  INT_8 =  7,
  UINT_8 = 8,
  INT_16 = 9,
  UINT_16 = 10,
  BOOL_8 = 11,
  INT_64 = 12,
  UINT_64 = 13
};

} // ns DlSystem


ALIAS_IN_ZDL_NAMESPACE(DlSystem, Runtime_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, RuntimeCheckOption_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, PerformanceProfile_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, ProfilingLevel_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, ExecutionPriorityHint_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, ImageEncoding_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, LogLevel_t)
ALIAS_IN_ZDL_NAMESPACE(DlSystem, IOBufferDataType_t)
