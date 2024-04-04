//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CHECKRUNTIME_H
#define CHECKRUNTIME_H

#include "SNPE/SNPEFactory.hpp"

zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime);
bool checkGLCLInteropSupport();

#endif
