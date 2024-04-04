// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_SESR_H
#define SUPERRESOLUTION_SESR_H

#include "Model.h"

class SESR: public Model {

public:
    void preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_SESR_H