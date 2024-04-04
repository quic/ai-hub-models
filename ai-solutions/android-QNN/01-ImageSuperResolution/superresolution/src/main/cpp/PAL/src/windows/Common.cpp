//==============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <Windows.h>
#include <io.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "Common.hpp"
#include "PAL/Debug.hpp"

int32_t pal::scanDir(const std::string &path, std::vector<WIN32_FIND_DATAA> &namelist) {
  // example : "C:/Users/guest" scan nothing, "C:/Users/guest/*" can scan the
  // entire directory instead
  std::string scanPath = path + "/*";
  WIN32_FIND_DATAA findFileData;
  HANDLE hFind = FindFirstFileA(scanPath.c_str(), &findFileData);
  if (hFind == INVALID_HANDLE_VALUE) {
    DEBUG_MSG("scanDir fail! Error code : %d", GetLastError());
    return -1;
  }

  do {
    // will compare char until '\0' to allow filename with first char = '.'
    if (strncmp(findFileData.cFileName, ".", 2) == 0 ||
        strncmp(findFileData.cFileName, "..", 3) == 0) {
      continue;
    }
    namelist.push_back(findFileData);
  } while (FindNextFileA(hFind, &findFileData));
  FindClose(hFind);

  return namelist.size();
}

void pal::normalizeSeparator(std::string &path) { replace(path.begin(), path.end(), '\\', '/'); }
