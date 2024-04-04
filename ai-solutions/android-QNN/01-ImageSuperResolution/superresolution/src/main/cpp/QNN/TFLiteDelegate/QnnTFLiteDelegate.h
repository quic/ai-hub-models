//==============================================================================
//
//  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifndef QNN_DELEGATE_CAPI_EXPORT
#define QNN_DELEGATE_CAPI_EXPORT
#endif /* QNN_DELEGATE_CAPI_EXPORT */

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Provide values to use for API version
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define QNN_DELEGATE_API_VERSION_MAJOR 0
#define QNN_DELEGATE_API_VERSION_MINOR 18
#define QNN_DELEGATE_API_VERSION_PATCH 0
// NOLINTEND(cppcoreguidelines-macro-usage)

/// A struct which is used to provide a version number using 3 values:
/// major, minor, patch
typedef struct {  // NOLINT(modernize-use-using)
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} QnnDelegateApiVersion;

/// The QNN backend used to delegate the model's nodes. Each backend has
/// its own set of supported ops and tensor types.
typedef enum TfLiteQnnDelegateBackendType {  // NOLINT(modernize-use-using)
  kUndefinedBackend = 0,
  /// Backend for Adreno<sup>TM</sup> GPU hardware accelerator.
  kGpuBackend,
  /// Backend for Hexagon HTP hardware accelerator.
  kHtpBackend,
  /// Backend for Hexagon DSP hardware accelerator.
  kDspBackend,
} TfLiteQnnDelegateBackendType;

/// Logging level of the delegate and QNN backend.
typedef enum TfLiteQnnDelegateLogLevel {  // NOLINT(modernize-use-using)
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
} TfLiteQnnDelegateLogLevel;

/// Options to set Graph Priority. This is directly mapped to Qnn_Priority_t.
/// Please refer to QNN SDK for additional information.
typedef enum TfLiteQnnDelegateGraphPriority {  // NOLINT(modernize-use-using)
  kQnnPriorityDefault = 0,
  kQnnPriorityLow,
  kQnnPriorityNormal,
  kQnnPriorityNormalHigh,
  kQnnPriorityHigh,
  kQnnPriorityUndefined,
} TfLiteQnnDelegateGraphPriority;

/// Options to profile the QNN Delegate execution.
typedef enum TfLiteQnnDelegateProfilingOptions {  // NOLINT(modernize-use-using)
  kProfilingOff = 0,
  kPerOpProfiling = 1,
} TfLiteQnnDelegateProfilingOptions;

/// Defines the optimization levels of the graph tensors that are not input
/// nor output tensors. This enum controls the trade-off between performance
/// and accuracy.
typedef enum TfLiteQnnDelegateGpuPrecision {  // NOLINT(modernize-use-using)
  kGpuUserProvided = 0,
  kGpuFp32,
  kGpuFp16,
  kGpuHybrid,
} TfLiteQnnDelegateGpuPrecision;

/// Defines performance modes available for GPU backend.
typedef enum TfLiteQnnDelegateGpuPerformanceMode {  // NOLINT(modernize-use-using)
  kGpuDefault = 0,
  kGpuHigh,
  kGpuNormal,
  kGpuLow,
} TfLiteQnnDelegateGpuPerformanceMode;

/// Defines performance modes available for HTP backend.
typedef enum TfLiteQnnDelegateHtpPerformanceMode {  // NOLINT(modernize-use-using)
  kHtpDefault = 0,
  kHtpSustainedHighPerformance = 1,
  kHtpBurst = 2,
  kHtpHighPerformance = 3,
  kHtpPowerSaver = 4,
  kHtpLowPowerSaver = 5,
  kHtpHighPowerSaver = 6,
  kHtpLowBalanced = 7,
  kHtpBalanced = 8,
} TfLiteQnnDelegateHtpPerformanceMode;

/// Defines performance modes available for DSP backend.
typedef enum TfLiteQnnDelegateDspPerformanceMode {  // NOLINT(modernize-use-using)
  kDspDefault = 0,
  kDspSustainedHighPerformance = 1,
  kDspBurst = 2,
  kDspHighPerformance = 3,
  kDspPowerSaver = 4,
  kDspLowPowerSaver = 5,
  kDspHighPowerSaver = 6,
  kDspLowBalanced = 7,
  kDspBalanced = 8,
} TfLiteQnnDelegateDspPerformanceMode;

///   Defines performance control strategy
///
///   **Manual**: The performance mode is voted as the backend is initialized,
///   and released at the moment of the backend is destroyed.
///
///   Users can control the vote/release of the performance mode by
///   TfLiteQnnDelegateSetPerf().
///
///   Note that this is the default strategy.
///
///   For example, users can vote before inferences, and release after all
///   inference done.
///
///   ~~~~~~~~~~~~~{.cpp}
///      TfLiteQnnDelegateSetPerf(delegate, kPerformanceVote);
///      // invoke inferences...
///      TfLiteQnnDelegateSetPerf(delegate, kPerformanceRelease);
///   ~~~~~~~~~~~~~
///
///   **AUTO**: QNN Delegate vote before an inference, and release after an idle
///   interval.
typedef enum TfLiteQnnDelegateHtpPerfCtrlStrategy {  // NOLINT(modernize-use-using)
  kHtpPerfCtrlManual = 0,
  kHtpPerfCtrlAuto = 1,
} TfLiteQnnDelegateHtpPerfCtrlStrategy;

/// Defines DSP performance control strategy. Similar to HTP cases.
typedef enum TfLiteQnnDelegateDspPerfCtrlStrategy {  // NOLINT(modernize-use-using)
  kDspPerfCtrlManual = 0,
  kDspPerfCtrlAuto = 1,
} TfLiteQnnDelegateDspPerfCtrlStrategy;

/// Defines pd sessions available for DSP backend.
typedef enum TfLiteQnnDelegateDspPdSession {  // NOLINT(modernize-use-using)
  kDspUnsignedPd = 0,
  kDspSignedPd,
  kDspAdaptivePd,
} TfLiteQnnDelegateDspPdSession;

/// Defines encoding for DSP backend. Dynamic encoding is more precise but
/// sacrifices a bit of performance.
typedef enum TfLiteQnnDelegateDspEncoding {  // NOLINT(modernize-use-using)
  kDspStatic = 0,
  kDspDynamic = 1,
  kDspUnknown = 0x7fffffff,
} TfLiteQnnDelegateDspEncoding;

/// Defines pd sessions available for HTP backend.
typedef enum TfLiteQnnDelegateHtpPdSession {  // NOLINT(modernize-use-using)
  kHtpUnsignedPd = 0,
  kHtpSignedPd,
} TfLiteQnnDelegateHtpPdSession;

/// Defines the optimization levels of the graph tensors that are not input nor
/// output tensors. This enum controls the trade-off between performance and
/// accuracy.
typedef enum TfLiteQnnDelegateHtpPrecision {  // NOLINT(modernize-use-using)
  kHtpQuantized = 0,
  kHtpFp16,
} TfLiteQnnDelegateHtpPrecision;

/// Defines the optimization strategy used by the HTP backend.
typedef enum TfLiteQnnDelegateHtpOptimizationStrategy {  // NOLINT(modernize-use-using)
  kHtpOptimizeForInference = 0,
  kHtpOptimizeForPrepare,
} TfLiteQnnDelegateHtpOptimizationStrategy;

/// Defines the performance action used by TfLiteQnnDelegateSetPerf()
typedef enum TfLiteQnnDelegatePerformanceAction {  // NOLINT(modernize-use-using)
  kPerformanceVote = 0,
  kPerformanceRelease = 1,
} TfLiteQnnDelegatePerformanceAction;

/// Specifies the backend options for the GPU backend. To be used when selecting
/// \ref TfLiteQnnDelegateBackendType.kGpuBackend for the \ref
/// TfLiteQnnDelegateOptions.backend_type.
typedef struct {  // NOLINT
  /// The default precision is half float for the best performance.
  TfLiteQnnDelegateGpuPrecision precision;
  /// The default performance mode sets high.
  TfLiteQnnDelegateGpuPerformanceMode performance_mode;
  /// The QNN GPU backend supports on-disk kernel persistence strategies where
  /// compiled GPU kernel binaries are cached to disk and can be shared across
  /// models having the same kernels and improve warm init times significantly.
  const char* kernel_repo_dir;
} TfLiteQnnDelegateGpuBackendOptions;

// clang-format off
#define QNN_DELEGATE_GPU_OPTION_INIT   \
  {                                   \
    kGpuFp16,    /*precision*/        \
    kGpuDefault, /*performance_mode*/ \
    ""           /*kernel_repo_dir*/  \
  }
// clang-format on

/// Specifies the backend options for the HTP backend. To be used when selecting
/// \ref TfLiteQnnDelegateBackendType.kGpuBackend for the \ref
/// TfLiteQnnDelegateOptions.backend_type.
typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the HTP.
  TfLiteQnnDelegateHtpPerformanceMode performance_mode;
  /// The default performance control strategy is Manual.
  TfLiteQnnDelegateHtpPerfCtrlStrategy perf_ctrl_strategy;
  /// The default precision mode supports quantized networks. Other precision
  /// modes may only be supported on certain SoCs.
  TfLiteQnnDelegateHtpPrecision precision;
  /// Signed or unsigned HTP PD session. The default PD session is unsigned.
  TfLiteQnnDelegateHtpPdSession pd_session;
  /// The default optimization strategy will optimize the graph for inference.
  TfLiteQnnDelegateHtpOptimizationStrategy optimization_strategy;
  /// With using short conv hmx, we might have better performance,
  /// but convolution that have short depth and/or weights that are not
  /// symmetric could exhibit inaccurate results.
  bool useConvHmx;
  /// With using fold relu, we might have better performance, this optimization
  /// is correct when quantization ranges for convolution are equal or subset of
  /// the Relu operation.
  bool useFoldRelu;
  /// Option to set VTCM size in MB. This is directly mapped to
  /// QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE under QnnHtpGraph_ConfigOption_t. If
  /// VTCM size is not set, the default VTCM size will be used.
  /// VTCM size must be bigger than 0, and if VTCM size is set to bigger than
  /// VTCM size available for this device, it will be set to the VTCM size
  /// available for this device.
  uint32_t vtcm_size;
  /// Option to set number of HVX threads. This is directly mapped to
  /// QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS under
  /// QnnHtpGraph_ConfigOption_t. If this this option is not set, the default
  /// number of hvx will be used. Input integer must be bigger than 0.
  /// If input exceeds the max number of HVX threads, it will clip to maximum
  /// number of threads supported.
  uint32_t num_hvx_threads;
} TfLiteQnnDelegateHtpBackendOptions;

// clang-format off
#define QNN_DELEGATE_HTP_OPTION_INIT                      \
  {                                                       \
    kHtpDefault,              /*performance_mode*/        \
    kHtpPerfCtrlManual,       /*perf_ctrl_strategy*/      \
    kHtpQuantized,            /*precision*/               \
    kHtpUnsignedPd,           /*pd_session*/              \
    kHtpOptimizeForInference, /*optimization_strategy*/   \
    false,                    /*useConvHmx*/              \
    false,                    /*useFoldRelu*/             \
    0,                        /*vtcm_size*/               \
    0,                       /*num_hvx_threads*/         \
  }
// clang-format on

/// Specifies the backend options for the DSP backend. To be used when selecting
/// kDspBackend as the <backend_type>.
typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the DSP.
  TfLiteQnnDelegateDspPerformanceMode performance_mode;
  /// The default performance control strategy is Manual.
  TfLiteQnnDelegateDspPerfCtrlStrategy perf_ctrl_strategy;
  /// The default PD session is unsigned.
  TfLiteQnnDelegateDspPdSession pd_session;
  /// The default Encoding is static
  TfLiteQnnDelegateDspEncoding encoding;
} TfLiteQnnDelegateDspBackendOptions;

// clang-format off
#define QNN_DELEGATE_DSP_OPTION_INIT                      \
  {                                                       \
    kDspDefault,              /*performance_mode*/        \
    kDspPerfCtrlManual,       /*perf_ctrl_strategy*/      \
    kDspUnsignedPd,           /*pd_session*/              \
    kDspStatic,               /*encoding*/              \
  }
// clang-format on

/// Map of TFLite custom operator name to op type defined within an op package.
typedef struct {  // NOLINT
  /// The TfLiteRegistration::custom_name set during registration.
  const char* custom_op_name;
  /// The corresponding op type name defined in the op package.
  const char* qnn_op_type_name;
} TfLiteQnnDelegateOpPackageOpMap;

// clang-format off
#define QNN_DELEGATE_OP_PACKAGE_OPTION_INIT   \
  {                                           \
    0,              /*num_op_package_infos*/  \
    nullptr,        /*op_package_infos*/      \
  }
// clang-format on

/// Structure containing the information needed to register and use an op
/// package with QNN.
typedef struct {  // NOLINT
  /// The name of the op package.
  const char* op_package_name;
  /// The path on disk to the op package library.
  const char* op_package_path;
  /// The name of a function in the op package library which satisfies the
  /// QnnOpPackage_InterfaceProvider_t interface.
  const char* interface_provider;
  /// The target which this op package library was compiled for.
  const char* target;
  /// Number of elements in the TfLiteQnnDelegateOpPackageInfo.ops_map array.
  int num_ops_map;
  /// An array of TfLiteQnnDelegateOpPackageOpMap structures.
  TfLiteQnnDelegateOpPackageOpMap* ops_map;
} TfLiteQnnDelegateOpPackageInfo;

typedef struct {  // NOLINT
  /// Number of elements in TfLiteQnnDelegateOpPackageOptions.op_package_infos
  /// array.
  int num_op_package_infos;
  /// An array of TfLiteQnnDelegateOpPackageInfo structures.
  TfLiteQnnDelegateOpPackageInfo* op_package_infos;
} TfLiteQnnDelegateOpPackageOptions;

typedef struct {  // NOLINT
  /// Set ops not to be delegated manually based on the op id(s).
  /// To obtain all the op ids, please refer to tensorflow/lite/builtin_ops.h.
  /// Notice that we skip *ALL* same type in \ref skip_delegate_ops array.
  /// For example, if you set skip
  /// SquaredDifference in your model, all of SquaredDifference ops in the
  /// model will not be delegated.
  const int* skip_delegate_ops;
  /// Indicate the length of \ref skip_delegate_ops array
  uint32_t skip_delegate_ops_nr;
  /// Set node not to be delegated manually based on the node id(s).
  /// Node id can be obtained by node's location information in .tflite
  const int* skip_delegate_node_ids;
  /// Indicate the length of \ref skip_delegate_node_ids array
  uint32_t skip_delegate_node_ids_nr;
} TfLiteQnnDelegateSkipOption;

// clang-format off
#define QNN_DELEGATE_SKIP_OPTION_INIT          \
  {                                            \
    nullptr,     /*skip_delegate_ops*/         \
    0,           /*skip_delegate_ops_nr*/      \
    nullptr,     /*skip_delegate_node_ids*/    \
    0,           /*skip_delegate_node_ids_nr*/ \
  }
// clang-format on

typedef struct {  // NOLINT
  /// The backend QNN library to open and execute the graph with. This is a
  /// required argument and will error out if kUndefinedBackend is supplied.
  TfLiteQnnDelegateBackendType backend_type;

  /// Optional parameter to override the QNN backend library.
  const char* library_path;

  /// Optional parameter specifying the directory of QNN Skel library. Only
  /// useful for backends which have a Skel library.
  const char* skel_library_dir;

  /// Optional backend specific options for the GPU backend. Only used when
  /// selecting \ref TfLiteQnnDelegateBackendType.kGpuBackend, else will be
  /// ignored.
  TfLiteQnnDelegateGpuBackendOptions gpu_options;

  /// Optional backend specific options for the HTP backend. Only used when
  /// selecting \ref TfLiteQnnDelegateBackendType.kHtpBackend, else will be
  /// ignored.
  TfLiteQnnDelegateHtpBackendOptions htp_options;

  /// Optional backend specific options for the DSP backend. Only used when
  /// selecting \ref TfLiteQnnDelegateBackendType.kDspBackend, else will be
  /// ignored.
  TfLiteQnnDelegateDspBackendOptions dsp_options;

  /// Logging level of the delegate and the backend. Default is off.
  TfLiteQnnDelegateLogLevel log_level;

  /// Option to enable profiling with the delegate. Default is off.
  TfLiteQnnDelegateProfilingOptions profiling;

  /// Optional structure to specify op packages loaded and used by the backend.
  TfLiteQnnDelegateOpPackageOptions op_package_options;

  /// Tensor dump output path. If a path is given, Delegate would write
  /// outputs of each OP there.
  /// In ALL cases, we don't recommend to set this option.
  /// This option exist just for debugging some accuracy issues.
  const char* tensor_dump_output_path;

  /// Specifies the directory of a compiled model.  Signals intent to either:
  ///   * Save the model if the file doesn't exist, or
  ///   * Restore model from the file.
  ///
  /// Model Cache specific options. Only used when setting \ref model_token,
  /// else will be ignored.
  ///
  /// At this moment, we recommend that delegate instances with/without cache
  /// should not be mixed in the same process, or at least an instance
  /// <b>without</b> cache is initialized, inferencing, and *terminate* before
  /// an instance with cache, in order to make sure all resources are prepared
  /// well.
  ///
  ///   ~~~~~~~~~~~~~{.cpp}
  ///
  ///   TfLiteDelegate* delegate_wo_cache =
  ///   TfLiteQnnDelegateCreate(&options_wo_cache);
  ///   interpreter_0->ModifyGraphWithDelegate(delegate_wo_cache);
  ///
  ///   // Perform inference with interpreter_0
  ///
  ///   TfLiteQnnDelegateDelete(delegate_wo_cache);
  ///
  ///   // after this, another delegate_with_cache can be used in the same
  ///   // process, though not recommended at this moment.
  ///  TfLiteDelegate* delegate_with_cache =
  ///  TfLiteQnnDelegateCreate(&options_with_cache);
  ///
  ///  // another interpreter
  ///  interpreter_1->ModifyGraphWithDelegate(delegate_with_cache);
  ///
  ///  // more delegates...etc.
  ///   ~~~~~~~~~~~~~
  const char* cache_dir;
  /// The unique null-terminated token string that acts as a ‘namespace’ for all
  /// serialization entries. Should be unique to a particular model (graph &
  /// constants). For an example of how to generate this from a TFLite model,
  /// see StrFingerprint() in lite/delegates/serialization.h.
  ///
  /// Model Cache specific options. Only used when setting \ref cache_dir, else
  /// will be ignored.
  const char* model_token;
  /// Option to skip node by specifying node types or node ids
  TfLiteQnnDelegateSkipOption skip_options;
  /// Option to set graph priority
  TfLiteQnnDelegateGraphPriority graph_priority;
} TfLiteQnnDelegateOptions;

// clang-format off
#define QNN_DELEGATE_OPTION_INIT                                        \
  {                                                                     \
    kUndefinedBackend,                    /*backend_type*/              \
    "",                                   /*library_path*/              \
    "",                                   /*skel_library_dir*/          \
    QNN_DELEGATE_GPU_OPTION_INIT,         /*gpu_options*/               \
    QNN_DELEGATE_HTP_OPTION_INIT,         /*htp_options*/               \
    QNN_DELEGATE_DSP_OPTION_INIT,         /*dsp_options*/               \
    kLogOff,                              /*log_level*/                 \
    kProfilingOff,                        /*profiling*/                 \
    QNN_DELEGATE_OP_PACKAGE_OPTION_INIT,  /*op_package_options*/        \
    "",                                   /*tensor_dump_output_path*/   \
    "",                                   /*cache_dir*/                 \
    "",                                   /*model_token*/               \
    QNN_DELEGATE_SKIP_OPTION_INIT,        /*skip_options*/              \
    kQnnPriorityDefault,                  /*graph_priority*/            \
  }
// clang-format on

typedef int32_t  // NOLINT(modernize-use-using)
    TfLiteQnnDelegateCapabilityStatus;

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
/// Return by TfLiteQnnDelegateHasCapability() if the capability is supported.
#define TfLiteQnnDelegateCapabilitySupported 1
/// Return by TfLiteQnnDelegateHasCapability() if the capability is not
/// supported.
#define TfLiteQnnDelegateCapabilityNotSupported 0
// NOLINTEND(cppcoreguidelines-macro-usage)

/// Defines possible QNN Delegate capabilities.
typedef enum TfLiteQnnDelegateCapability {  // NOLINT(modernize-use-using)
  kCapHtpRuntimeQuant = 0,
  kCapHtpRuntimeFp16 = 1,
  kCapGpuRuntime = 2,
  kCapDspRuntime = 3,
} TfLiteQnnDelegateCapability;

/// Create the QNN Delegate options structure and populate with default values.
QNN_DELEGATE_CAPI_EXPORT TfLiteQnnDelegateOptions
TfLiteQnnDelegateOptionsDefault();

/// Create the QNN Delegate with the specified options.
QNN_DELEGATE_CAPI_EXPORT TfLiteDelegate* TfLiteQnnDelegateCreate(
    const TfLiteQnnDelegateOptions* options);

/// Delete the QNN Delegate once no longer required.
QNN_DELEGATE_CAPI_EXPORT void TfLiteQnnDelegateDelete(TfLiteDelegate* delegate);

/// Manually vote or release performance mode. "Vote" to request hardware to
/// obey the performance mode setting as it can as possible. "Release" to
/// release the vote. Note that this API only work for HTP/DSP backend with \ref
/// kHtpPerfCtrlManual or \ref kDspPerfCtrlManual. Return true for success,
/// false for failure.
QNN_DELEGATE_CAPI_EXPORT bool TfLiteQnnDelegateSetPerf(
    TfLiteDelegate* delegate, const TfLiteQnnDelegatePerformanceAction action);

/// Detect whether the capability is supported on the platform running QNN
/// Delegate.
///
/// Note that this is an experimental feature.
QNN_DELEGATE_CAPI_EXPORT TfLiteQnnDelegateCapabilityStatus
TfLiteQnnDelegateHasCapability(const TfLiteQnnDelegateCapability cap);

/// This API changes the performance mode of a created QNN Delegate on HTP
/// backend, returning `true` for the mode set correctly, `false` for any
/// failure.
///
/// It will perform a vote after a successful update. If the strategy of
/// performance controling is **Manaul**, the new mode takes effect before this
/// API returns.
///
/// Note that this API cannot be called during inferecing, and this is an
/// experimental feature.
QNN_DELEGATE_CAPI_EXPORT bool TfLiteQnnDelegateUpdateDspPerfMode(
    TfLiteDelegate* delegate, const TfLiteQnnDelegateDspPerformanceMode mode);

/// This API changes the performance mode of a created QNN Delegate on DSP
/// backend, returning `true` for the mode set correctly, `false` for any
/// failure.
QNN_DELEGATE_CAPI_EXPORT bool TfLiteQnnDelegateUpdateHtpPerfMode(
    TfLiteDelegate* delegate, const TfLiteQnnDelegateHtpPerformanceMode mode);

/// Get QNN Delegate API version.
QNN_DELEGATE_CAPI_EXPORT QnnDelegateApiVersion TfLiteQnnDelegateGetApiVersion();

/// Allocate specific tensors (usually graph inputs and outputs) on shared
/// memory. Users are responsible to allocate "enough" tensor bytes, and set
/// alignment as kDefaultTensorAlignment. The fuction returns a valid pointer if
/// allocation is successful.
///
/// Note that this is an experimental feature.
QNN_DELEGATE_CAPI_EXPORT void* TfLiteQnnDelegateAllocCustomMem(
    size_t bytes, size_t alignment);

/// Free the allocated shared memory.
///
/// Note that this is an experimental feature.
QNN_DELEGATE_CAPI_EXPORT void TfLiteQnnDelegateFreeCustomMem(void* buffer_ptr);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_
