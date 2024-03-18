// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imageclassification;
import java.util.List;

public class Result<E> {

    private final E results;
    private final long inferenceTime;
    public Result(E results, long inferenceTime) {

        this.results = results;
        this.inferenceTime = inferenceTime;
    }

    public E getResults() {
        return results;
    }


    public long getInferenceTime() {
        return inferenceTime;
    }

}
