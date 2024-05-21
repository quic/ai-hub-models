// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.qcom.aihub_segmentation;

//Singleton class to maintain TFLiteModelExecutor Object
public class ModelManager {
    private static ModelManager instance;
    private TFLiteModelExecutor inferObj;

    private ModelManager() {
        // Private constructor to prevent instantiation
    }

    public static ModelManager getInstance() {
        if (instance == null) {
            instance = new ModelManager();
        }
        return instance;
    }

    public TFLiteModelExecutor getModelExecutor() {
        return inferObj;
    }

    public void initializeModelExecutor(TFLiteModelExecutor TFLiteModelExecutor) {
        this.inferObj = TFLiteModelExecutor;
    }
}
