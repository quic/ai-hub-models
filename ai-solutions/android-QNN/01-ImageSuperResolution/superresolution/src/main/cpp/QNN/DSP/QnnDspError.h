//=============================================================================
//
//  Copyright (c) 2020-2021, 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#ifndef QNN_DSP_ERROR_H
#define QNN_DSP_ERROR_H

#include "QnnError.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  QNN_DSP_ERROR_MIN_ERROR = QNN_MIN_ERROR_ERROR,
  ////////////////////////////////////////

  /// Qnn Error success
  QNN_DSP_ERROR_NO_ERROR = QNN_SUCCESS,
  /// Error manager not initialized
  QNN_DSP_ERROR_NOT_INITIALIZED = QNN_MIN_ERROR_ERROR + 0,
  /// Invalid error manager config
  QNN_DSP_ERROR_INVALID_CONFIG = QNN_MIN_ERROR_ERROR + 1,
  /// Invalid reporting level in error manager config
  QNN_DSP_ERROR_INVALID_REPORTING_LEVEL = QNN_MIN_ERROR_ERROR + 2,
  /// Error handle not recognized
  QNN_DSP_ERROR_INVALID_ERRORHANDLE = QNN_MIN_ERROR_ERROR + 3,
  /// Invalid storage limit in error manger config
  QNN_DSP_ERROR_INVALID_STORAGE_LIMIT = QNN_MIN_ERROR_ERROR + 4,
  /// Invalid context identifier in error handle
  QNN_DSP_ERROR_INVALID_CONTEXTHANDLE = QNN_MIN_ERROR_ERROR + 5,
  /// Error info not found
  QNN_DSP_ERROR_MISSING_ERRORINFO = QNN_MIN_ERROR_ERROR + 6,
  /// Invalid function argument
  QNN_DSP_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_ERROR + 7,
  /// No space to accept new errors
  QNN_DSP_ERROR_OUT_OF_MEMORY = QNN_MIN_ERROR_ERROR + 8,
  /// Error manager already contains maximum number of contexts
  QNN_DSP_ERROR_MAX_NUM_CONTEXTS = QNN_MIN_ERROR_ERROR + 9,
  /// Unknown error
  QNN_DSP_ERROR_UNKNOWN_ERROR = QNN_MIN_ERROR_ERROR + 10,

  ////////////////////////////////////////
  QNN_DSP_ERROR_MAX_ERROR = QNN_MAX_ERROR_ERROR,
  // Unused, present to ensure 32 bits.
  QNN_DSP_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnDspError_Error_t;

/**
 * @brief Struct which defines static error info
 */
typedef struct {
  /// 16-bit Error code extracted from error handle
  uint16_t errorCode;
} QnnDspError_StaticInfo_t;

/**
 * @brief Struct which defines detailed error info
 */
typedef struct {
  QnnDspError_StaticInfo_t staticInfo;
  uint32_t errorBufSize;
  void *errorBuffer;
} QnnDspError_Blob_t;

#define QNN_DSP_ERROR_MAX_STRING_LEN 256

/**
 * @brief Maximum Storage allowed per context
 *        by DSP backend for saving errors
 *
 * @note The storage limit passed in QnnBackend_Config_t
 *       during QnnBackend_initialize and in QnnContext_Config_t
 *       during QnnContext_create should be less than or equal to
 *       this allowed storage
 */
#define QNN_DSP_ERROR_MAX_CONTEXT_STORAGE 2

/**
 * @brief This struct is used to provide
 *        verbose error string
 *
 * @note
 *       - This struct is added to QnnDspError_Info_t
 *         from which the errorString can be accessed by the
 *         user for errorType - QNN_DSP_ERROR_TYPE_VERBOSE
 *       - This is a C string - It is null terminated
 */
typedef struct {
  char errorString[QNN_DSP_ERROR_MAX_STRING_LEN];
} QnnDspError_Verbose_t;

/// QnnDspError_Verbose_t initializer macro
#define QNN_DSP_ERROR_VERBOSE_INIT \
  {                                \
    { 0 } /*errorString*/          \
  }

typedef enum {
  QNN_DSP_ERROR_TYPE_VERBOSE,
  // to ensure 32 bit
  QNN_DSP_ERROR_TYPE_MAX = 0x7FFFFFFF
} QnnDspError_Type_t;

/**
 * @brief        This is the error info struct provided as errorBuffer in
 *               QnnDspError_Blob_t
 *
 *               The errorInfo can be obtained by casting errorBuffer field
 *               of QnnDspError_Blob_t to QnnDspError_Info_t.
 *               Based on the errorType, the type of error info structure can be found
 *               Below is the Map between QnnDspError_Type_t and type of Error Info Struct
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+----------------------------+-------------------------+
 *               | #  | errorType                  | errorInfo struct type   |
 *               +====+============================+=========================+
 *               | 1  | QNN_DSP_ERROR_TYPE_VERBOSE | QnnDspError_Verbose_t   |
 *               +----+----------------------------+-------------------------+
 *               \endverbatim
 */
typedef struct {
  QnnDspError_Type_t errorType;
  union {
    QnnDspError_Verbose_t verboseInfo;
  };
} QnnDspError_Info_t;

/// QnnDspError_Info_t initializer macro
#define QNN_DSP_ERROR_INFO_INIT                  \
  {                                              \
    QNN_DSP_ERROR_TYPE_MAX, /*errorType*/        \
    {                                            \
      QNN_DSP_ERROR_VERBOSE_INIT /*verboseInfo*/ \
    }                                            \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_DSP_ERROR_H
