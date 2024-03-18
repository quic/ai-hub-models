// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.imagesuperres;
import java.util.List;

public class Result<E> {

    private final E results;
    private final long inferenceTime;
    private final String remarks;
    private boolean status = false;
    public Result(E results, long inferenceTime,String remarks) {

        this.results = results;
        this.inferenceTime = inferenceTime;
        this.remarks = remarks;


        if(inferenceTime>0) this.status = true;
    }

    public E getResults() {
        return results;
    }

    public String getRemarks() {
        return remarks;
    }

    public long getInferenceTime() {
        return inferenceTime;
    }

    public boolean getStatus(){return status; }

}
