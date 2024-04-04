//==============================================================================
//
// Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// clang-format off
#include <stdlib.h>
#include <windows.h>
#include <libloaderapi.h>
#include <psapi.h>
#include <winevt.h>
// clang-format on

#include <set>
#include <string>

#include "PAL/Debug.hpp"
#include "PAL/DynamicLoading.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x)  STRINGIFY(x)

static std::set<HMODULE> mod_handles;
static thread_local char *sg_lastErrMsg = "";

void *pal::dynamicloading::dlOpen(const char *filename, int flags) {
  HMODULE mod;
  HANDLE cur_proc;
  DWORD as_is, to_be;
  bool loadedBefore = false;

  if (!filename || ::strlen(filename) == 0) {
    // TODO: we don't support empty filename now
    sg_lastErrMsg = "filename is null or empty";
    return NULL;
  }

  // POSIX asks one of symbol resolving approaches:
  // NOW or LAZY must be specified
  if (!(flags & DL_NOW)) {
    // TODO: since Windows does not provide existing API so lazy
    // symbol resolving needs to do relocation by ourself
    // that would be too costly. SNPE didn't use this feature now
    // , wait until we really need it. keep the flexibility here
    // ask caller MUST pass DL_NOW
    sg_lastErrMsg = "flags must include DL_NOW";
    return NULL;
  }

  cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, NULL, 0, &as_is) == 0) {
    sg_lastErrMsg = "enumerate modules failed before loading module";
    return NULL;
  }

  // search from system lib path first
  mod = LoadLibraryExA(filename, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!mod) {
    sg_lastErrMsg = "load library failed";
    return NULL;
  }

  if (EnumProcessModules(cur_proc, NULL, 0, &to_be) == 0) {
    sg_lastErrMsg = "enumerate modules failed after loading module";
    FreeLibrary(mod);
    return NULL;
  }

  if (as_is == to_be) {
    loadedBefore = true;
  }

  // (not loadedBefore) and DL_LOCAL means this lib was not loaded yet
  // add it into the local set
  //
  // If loadedBefore and DL_LOCAL, means this lib was already loaded
  // 2 cases here for how it was loaded before:
  // a. with DL_LOCAL, just ignore since it was already in local set
  // b. with DL_GLOBAL, POSIX asks it in global, ignore it, too
  if ((!loadedBefore) && (flags & DL_LOCAL)) {
    mod_handles.insert(mod);
  }

  // once callers ask for global, needs to be in global thereafter
  // so the lib should be removed from local set
  if (flags & DL_GLOBAL) {
    mod_handles.erase(mod);
  }

  return static_cast<void *>(mod);
}

void *pal::dynamicloading::dlSym(void *handle, const char *symbol) {
  FARPROC sym_addr = NULL;
  HANDLE cur_proc;
  DWORD size, size_needed;
  HMODULE *mod_list;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return NULL;
  }

  cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, NULL, 0, &size) == 0) {
    sg_lastErrMsg = "enumerate modules failed before memory allocation";
    return NULL;
  }

  mod_list = static_cast<HMODULE *>(malloc(size));
  if (!mod_list) {
    sg_lastErrMsg = "malloc failed";
    return NULL;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    sg_lastErrMsg = "enumerate modules failed after memory allocation";
    free(mod_list);
    return NULL;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles.find(mod_list[i]);
      if (iter != mod_handles.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void **)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    sg_lastErrMsg = "can't resolve symbol";
    return NULL;
  }

  return *(void **)(&sym_addr);
}

int pal::dynamicloading::dlAddrToLibName(void *addr, std::string &name) {
  // Clean the output buffer
  name = std::string();

  // If the address is empty, return zero as treating failure
  if (!addr) {
    DEBUG_MSG("Input address is nullptr.");
    return 0;
  }

  HMODULE hModule = NULL;
  // TODO: Need to use TCHAR for the compatibility of ASCII and Unicode
  CHAR nameBuf[MAX_PATH];

  // (1st flag) The lpModuleName parameter is an address in the module
  // (2nd flag) The reference count for the module is not incremented
  DWORD flags =
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;

  // Retrieves a module handle for the specified module by its symbol address
  if (!GetModuleHandleExA(flags, reinterpret_cast<LPCSTR>(addr), &hModule) || hModule == NULL) {
    DEBUG_MSG("Failed to get module handle. Error code: %d", GetLastError());
    return 0;
  }

  // Retrieves the fully qualified path for the file that contains the specified module
  DWORD dwSize = GetModuleFileNameA(hModule, nameBuf, sizeof(nameBuf));

  // dwSize == 0 indicates function failure
  // If the path is too long (greater than MAX_PATH), treat it as failure
  if (dwSize == 0 || ERROR_INSUFFICIENT_BUFFER == GetLastError()) {
    DEBUG_MSG("Failed to get module file name. Error code: %d", GetLastError());
    return 0;
  }

  name = std::string(nameBuf);

  // Return a non-zero value to represent the function successes
  return 1;
}

int pal::dynamicloading::dlClose(void *handle) {
  if (!handle) {
    return 0;
  }

  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    sg_lastErrMsg = "free library failed";
    return -1;
  }

  mod_handles.erase(mod);

  return 0;
}

char *pal::dynamicloading::dlError(void) {
  char *retStr = sg_lastErrMsg;

  sg_lastErrMsg = "";

  return retStr;
}
