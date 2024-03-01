// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef _SNPERUNTIME_H_
#define _SNPERUNTIME_H_

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlContainer/IDlContainer.hpp"

#include "Utils.h"

namespace snperuntime {

class SNPERuntime {
public:
    SNPERuntime();

    bool Initialize(const std::string& model_path, const runtime_t runtime);
    bool Deinitialize();
    bool SetOutputLayers(std::vector<std::string>& outputLayers);

    std::vector<size_t> GetInputShape(const std::string& name);
    std::vector<size_t> GetOutputShape(const std::string& name);

    float* GetInputTensor(const std::string& name);
    float* GetOutputTensor(const std::string& name);

    bool IsInit() {
        return m_isInit;
    }

    bool execute();

private:
    bool m_isInit = false;

    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    zdl::DlSystem::Runtime_t m_runtime;
    zdl::DlSystem::StringList m_outputLayers;

    std::map<std::string, std::vector<size_t> > m_inputShapes;
    std::map<std::string, std::vector<size_t> > m_outputShapes;

    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_inputUserBuffers;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_outputUserBuffers;
    zdl::DlSystem::UserBufferMap m_inputUserBufferMap;
    zdl::DlSystem::UserBufferMap m_outputUserBufferMap;
    zdl::DlSystem::PerformanceProfile_t m_profile;

    void setTargetRuntime(const runtime_t runtime);
    void setPerformanceProfile(const performance_t perfprofile);

    std::unordered_map<std::string, std::vector<float>> m_applicationInputBuffers;
    std::unordered_map<std::string, std::vector<float>> m_applicationOutputBuffers;
};

}

#endif // _SNPERUNTIME_H_