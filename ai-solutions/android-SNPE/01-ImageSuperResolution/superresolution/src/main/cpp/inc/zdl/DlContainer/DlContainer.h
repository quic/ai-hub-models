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

#ifndef DL_CONTAINER_DLCONTAINER_H
#define DL_CONTAINER_DLCONTAINER_H

#ifdef __cplusplus
#include <cstdint> // uint8_t
#include <cstddef> // size_t
#else
#include <stdint.h>
#include <stddef.h>
#endif

#include "DlSystem/DlError.h"
#include "DlSystem/StringList.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A typedef to indicate a SNPE DlcRecord handle
 */
typedef void* Snpe_DlcRecord_Handle_t;

/**
 * Constructs a DlcRecord and returns a handle to it
 *
 * @return the handle to the created DlcRecord
 */
SNPE_API
Snpe_DlcRecord_Handle_t Snpe_DlcRecord_Create();

/**
 * Constructs a DlcRecord with a provided name and returns a handle to it
 *
 * @param[in] name : the name of the record
 *
 * @return the handle to the created DlcRecord
 */
SNPE_API
Snpe_DlcRecord_Handle_t Snpe_DlcRecord_CreateName(const char* name);


/**
 * Destroys/frees a DlcRecord
 *
 * @param[in] dlcRecordHandle : Handle to access DlcRecord
 *
 * @return indication of success/failures
 */
SNPE_API
Snpe_ErrorCode_t Snpe_DlcRecord_Delete(Snpe_DlcRecord_Handle_t dlcRecordHandle);

/**
 * Gets the size of a DlcRecord in bytes
 *
 * @param[in] dlcRecordHandle : Handle to access DlcRecord
 *
 * @return the size of the DlcRecord in bytes
 */
SNPE_API
size_t Snpe_DlcRecord_Size(Snpe_DlcRecord_Handle_t dlcRecordHandle);

/**
 * Gets a pointer to the start of the DlcRecord's data
 *
 * @param[in] dlcRecordHandle : Handle to access DlcRecord
 *
 * @return uint8_t pointer to the DlcRecord's data
 */
SNPE_API
uint8_t* Snpe_DlcRecord_Data(Snpe_DlcRecord_Handle_t dlcRecordHandle);

/**
 * Gets the name of the DlcRecord
 *
 * @param[in] dlcRecordHandle : Handle to access DlcRecord
 *
 * @return the record's name
 */
SNPE_API
const char* Snpe_DlcRecord_Name(Snpe_DlcRecord_Handle_t dlcRecordHandle);

/**
 * A typedef to indicate a SNPE DlContainer handle
 */
typedef void* Snpe_DlContainer_Handle_t;

/**
 * Destroys/frees a DlContainer
 *
 * @param[in] dlContainerHandle : Handle to access DlContainer
 *
 * @return indication of success/failures
 */
SNPE_API
Snpe_ErrorCode_t Snpe_DlContainer_Delete(Snpe_DlContainer_Handle_t dlContainerHandle);


/**
 * Initializes a container from a container archive file.
 *
 * @param[in] filename Container archive file path.
 *
 * @return Status of container open call
 */
SNPE_API
Snpe_DlContainer_Handle_t Snpe_DlContainer_Open(const char* filename);

/**
  * Initializes a container from a byte buffer.
  *
  * @param[in] buffer Byte buffer holding the contents of an archive
  *                   file.
  *
  * @param[in] size Size of the byte buffer.
  *
  * @return A Snpe_DlContainer_Handle_t to access the dlContainer
 */
SNPE_API
Snpe_DlContainer_Handle_t Snpe_DlContainer_OpenBuffer(const uint8_t* buffer, const size_t size);

/**
 * Get the record catalog for a container.
 *
 * @param[in] dlContainerHandle : Handle to access DlContainer
 *
 * @return A Snpe_StringListHandle_t that holds the record names of the DlContainer
 */
SNPE_API
Snpe_StringList_Handle_t Snpe_DlContainer_GetCatalog(Snpe_DlContainer_Handle_t dlContainerHandle);

/**
 * Get a record from a container by name.
 *
 * @param[in] dlContainerHandle : Handle to access DlContainer
 * @param[in] recordName : Name of the record to fetch.
 *
 * @return A Snpe_DlcRecordHandle_t that owns the record read from the DlContainer
 */
SNPE_API
Snpe_DlcRecord_Handle_t Snpe_DlContainer_GetRecord(Snpe_DlContainer_Handle_t dlContainerHandle, const char* recordName);

/**
 * Save the container to an archive on disk. This function will save the
 * container if the filename is different from the file that it was opened
 * from, or if at least one record was modified since the container was
 * opened.
 *
 * It will truncate any existing file at the target path.
 *
 * @param[in] dlContainerHandle : Handle to access DlContainer
 * @param[in] filename : Container archive file path.
 *
 * @return indication of success/failure
 */
SNPE_API
Snpe_ErrorCode_t Snpe_DlContainer_Save(Snpe_DlContainer_Handle_t dlContainerHandle, const char* filename);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // DL_CONTAINER_DLCONTAINER_H
