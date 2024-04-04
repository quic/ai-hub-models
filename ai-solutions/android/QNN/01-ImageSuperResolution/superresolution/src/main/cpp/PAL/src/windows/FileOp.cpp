//==============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <limits.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/locking.h>
#include <windows.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Common.hpp"
#include "PAL/Debug.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

//-------------------------------------------------------------------------------
//    pal::FileOp::checkFileExists
//-------------------------------------------------------------------------------
bool pal::FileOp::checkFileExists(const std::string &fileName) {
  struct stat st;
  if (stat(fileName.c_str(), &st) != 0) {
    DEBUG_MSG("Check File fail! Error code : %d", errno);
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::copyOverFile
//-------------------------------------------------------------------------------
bool pal::FileOp::copyOverFile(const std::string &fromFile, const std::string &toFile) {
  if (CopyFileA(fromFile.c_str(), toFile.c_str(), 0) == 0) {
    DEBUG_MSG("Copy file fail! Error code : %d", GetLastError());
    return false;
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::move
//-------------------------------------------------------------------------------
bool pal::FileOp::move(const std::string &currentName, const std::string &newName, bool overwrite) {
  struct stat st;
  // if currentName doesn't exist, return false in case newName got deleted
  if (stat(currentName.c_str(), &st) != 0) {
    DEBUG_MSG("CurrentName check status fail! Error code : %d", errno);
    return false;
  }
  if (stat(newName.c_str(), &st) == 0) {
    if ((st.st_mode & S_IFDIR) != 0) {
      // if newName is directory and overwrite = false, cannot move, return false
      // if newName is directory and overwrite = true, delete it and rename
      if (overwrite == false) {
        return false;
      }
      pal::Directory::remove(newName);
    } else {
      deleteFile(newName);
    }
  }
  // in windows, if newName exist already, rename will return -1
  // only when newName doesn't exist, rename will return 0
  return (rename(currentName.c_str(), newName.c_str()) == 0);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::deleteFile
//-------------------------------------------------------------------------------
bool pal::FileOp::deleteFile(const std::string &fileName) {
  return (DeleteFileA(fileName.c_str()) != 0);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::checkIsDir
//-------------------------------------------------------------------------------
bool pal::FileOp::checkIsDir(const std::string &fileName) {
  DWORD result = GetFileAttributesA(fileName.c_str());
  if (result == static_cast<DWORD>(FILE_INVALID_FILE_ID)) {
    DEBUG_MSG("File attribute is invalid_file_id!");
    return false;
  }
  return (result & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfo
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfo(const std::string &filename,
                              pal::FileOp::FilenamePartsType_t &filenameParts) {
  std::string name;
  int32_t lastPathSeparator = std::max(static_cast<int32_t>(filename.find_last_of('\\')),
                                       static_cast<int32_t>(filename.find_last_of('/')));
  if (lastPathSeparator == static_cast<int32_t>(std::string::npos)) {
    // No directory
    name = filename;
  } else {
    // has a directory part
    filenameParts.directory = filename.substr(0, lastPathSeparator);
    name                    = filename.substr(lastPathSeparator + 1);
  }

  size_t ext = name.find_last_of(".");
  if (ext == std::string::npos) {
    // no extension
    filenameParts.basename = name;
  } else {
    // has extension
    filenameParts.basename  = name.substr(0, ext);
    filenameParts.extension = name.substr(ext + 1);
  }
  pal::normalizeSeparator(filenameParts.directory);
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursiveImpl
//-------------------------------------------------------------------------------
static bool getFileInfoListRecursiveImpl(const std::string &path,
                                         pal::FileOp::FilenamePartsListType_t &filenamePartsList,
                                         const bool ignoreDirs,
                                         size_t maxDepth) {
  // base case
  if (maxDepth == 0) {
    return true;
  }
  if (pal::FileOp::checkIsDir(path) == false) {
    return false;
  }
  int32_t entryCount = 0;
  std::vector<WIN32_FIND_DATAA> nameList;
  entryCount = pal::scanDir(path.c_str(), nameList);
  if (entryCount < 0) {
    return false;
  }
  while (entryCount--) {
    const std::string dName = std::string(nameList[entryCount].cFileName);
    // skip current directory, previous directory and empty string
    if (dName.empty() || dName == "." || dName == "..") {
      continue;
    }
    std::string curPath = path + pal::Path::getSeparator() + dName;
    // recursive if directory but avoid symbolic links to directories
    if (pal::FileOp::checkIsDir(curPath)) {
      struct stat st;
      if (stat(curPath.c_str(), &st) == 0 && ((st.st_mode & S_IFDIR) != 0) &&
          (!getFileInfoListRecursiveImpl(curPath, filenamePartsList, ignoreDirs, maxDepth - 1))) {
        return false;
      }
      if (curPath.back() != pal::Path::getSeparator()) {
        curPath += pal::Path::getSeparator();
      }
      // continue here to prevent this object from adding filenameparts in
      // vector but we still need this directory to go recursive
      if (ignoreDirs) {
        continue;
      }
    }
    // add to vector
    pal::FileOp::FilenamePartsType_t filenameParts = {std::string(), std::string(), std::string()};
    if (pal::FileOp::getFileInfo(curPath, filenameParts)) {
      filenamePartsList.push_back(filenameParts);
    }
  }
  return true;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoList
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfoList(const std::string &path,
                                  FilenamePartsListType_t &filenamePartsList) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, false, 1);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursive
//-------------------------------------------------------------------------------
bool pal::FileOp::getFileInfoListRecursive(const std::string &path,
                                           FilenamePartsListType_t &filenamePartsList,
                                           const bool ignoreDirs) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, ignoreDirs, UINT_MAX);
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getAbsolutePath
//-------------------------------------------------------------------------------
std::string pal::FileOp::getAbsolutePath(const std::string &path) {
  char fullPath[MAX_PATH];
  if (_fullpath(fullPath, path.c_str(), MAX_PATH) == NULL) {
    DEBUG_MSG("GetAbsolute path fail! Error code : %d", errno);
    return std::string();
  }
  std::string reStr = std::string(fullPath);
  pal::normalizeSeparator(reStr);
  return reStr;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getDirectory
//-------------------------------------------------------------------------------
std::string pal::FileOp::getDirectory(const std::string &file) {
  std::string rc = file;
  int32_t index  = std::max(static_cast<int32_t>(file.find_last_of('\\')),
                           static_cast<int32_t>(file.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = file.substr(0, index);
  }
  pal::normalizeSeparator(rc);
  return rc;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::GetFileName
//-------------------------------------------------------------------------------
std::string pal::FileOp::getFileName(const std::string &file) {
  std::string rc = file;
  int32_t index  = std::max(static_cast<int32_t>(file.find_last_of('\\')),
                           static_cast<int32_t>(file.find_last_of('/')));
  if (index != static_cast<int32_t>(std::string::npos)) {
    rc = file.substr(index + 1);  // +1 to skip path separator
  }
  return rc;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::hasFileExtension
//-------------------------------------------------------------------------------
bool pal::FileOp::hasFileExtension(const std::string &file) {
  FilenamePartsType_t parts = {std::string(), std::string(), std::string()};
  getFileInfo(file, parts);
  return !parts.extension.empty();
}

//-------------------------------------------------------------------------------
//    pal::FileOp::getCurrentWorkingDirectory
//-------------------------------------------------------------------------------
std::string pal::FileOp::getCurrentWorkingDirectory() {
  char buffer[MAX_PATH + 1];
  buffer[0] = '\0';

  // If there is any failure return empty string. It is technically possible
  // to handle paths exceeding PATH_MAX on some flavors of *nix but platforms
  // like Android (Bionic) do no provide such capability. For consistency we
  // will not handle extra long path names.
  if (0 == GetCurrentDirectoryA(MAX_PATH, buffer)) {
    DEBUG_MSG("Get current working directory fail! Error code : %d", GetLastError());
    return std::string();
  }
  std::string res = std::string(buffer);
  pal::normalizeSeparator(res);
  return res;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::setCurrentWorkingDirectory
//-------------------------------------------------------------------------------
bool pal::FileOp::setCurrentWorkingDirectory(const std::string &workingDir) {
  return _chdir(workingDir.c_str()) == 0;
}

//-------------------------------------------------------------------------------
//    pal::FileOp::PartsToString
//-------------------------------------------------------------------------------
std::string pal::FileOp::partsToString(const FilenamePartsType_t &filenameParts) {
  std::string path;

  if (!filenameParts.directory.empty()) {
    path += filenameParts.directory;
    path += Path::getSeparator();
  }
  if (!filenameParts.basename.empty()) {
    path += filenameParts.basename;
  }
  if (!filenameParts.extension.empty()) {
    path += ".";
    path += filenameParts.extension;
  }
  pal::normalizeSeparator(path);
  return path;
}