//==============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <Windows.h>
#include <io.h>

#include <iostream>
#include <vector>

namespace pal {
/**
 * @brief
 *   Scans elements in a directory.
 * @param path
 *   Path in string which we are going to scan.
 * @param namelist
 *   Data struct for each element, which will be stored as WIN32_FIND_DATAA.
 * @return
 *   Number of elements in this path, return -1 if fail.
 */
int32_t scanDir(const std::string &path, std::vector<WIN32_FIND_DATAA> &namelist);

/**
 * @brief
 *   Replace all the '\\' in path with '/' to keep consistency.
 * @param path
 *   The string which you want to format.
 */
void normalizeSeparator(std::string &path);
}  // namespace pal
