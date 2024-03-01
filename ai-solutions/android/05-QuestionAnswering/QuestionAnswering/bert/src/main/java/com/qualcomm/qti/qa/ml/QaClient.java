// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@\n
// =============================================================================

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Changes from QuIC are provided under the following license:

Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

SPDX-License-Identifier: BSD-3-Clause
==============================================================================*/

package com.qualcomm.qti.qa.ml;

import android.content.Context;
import android.content.res.AssetManager;
import androidx.annotation.WorkerThread;
import android.util.Log;
import android.widget.Toast;

import com.google.common.base.Joiner;
import java.io.BufferedReader;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Interface to load SNPE model and provide predictions. */
public class QaClient {
  private static final String TAG = "SNPE_Client";
  private static final String DIC_PATH = "vocab.txt";

  private static final int MAX_ANS_LEN = 32;
  private static final int MAX_QUERY_LEN = 64;
  private static final int MAX_SEQ_LEN = 384;
  private static final boolean DO_LOWER_CASE = true;
  private static final int PREDICT_ANS_NUM = 3;// default 5; can be set to 3 without issues
  private static final int NUM_LITE_THREADS = 1;

  // flag to track if SNPE instance is initialized
  private static boolean doSnpeInit = true;

  // Need to shift 1 for outputs ([CLS]).
  private static final int OUTPUT_OFFSET = 1;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final FeatureConverter featureConverter;
  private AssetManager assetManager;

  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  static {
    System.loadLibrary("qa");
  }

  public QaClient(Context context) {
    this.context = context;
    this.featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
  }

  @WorkerThread
  public synchronized String loadModel(String Model) {
    String uiLogger = "";
    try {
      // query runtimes & init SNPE
      if (doSnpeInit) {
        String nativeDirPath = context.getApplicationInfo().nativeLibraryDir;

        uiLogger += queryRuntimes(nativeDirPath);

        // init SNPE
        assetManager = context.getAssets();
        Toast.makeText(context,"Initializing SNPE",Toast.LENGTH_SHORT).show();
        Log.i(TAG, "onCreate: Initializing SNPE ...");
        uiLogger = initSNPE(assetManager, Model);

        doSnpeInit = false;
      }
    } catch (Exception ex) {
      Log.e(TAG, ex.getMessage());
      uiLogger += ex.getMessage();
    }
    return uiLogger;
  }

  @WorkerThread
  public synchronized void loadDictionary() {
    try {
      loadDictionaryFile(this.context.getAssets());
      Log.v(TAG, "Dictionary loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized void unload() {
    dic.clear();
  }

  /** Load dictionary from assets. */
  public void loadDictionaryFile(AssetManager assetManager) throws IOException {
    try (InputStream ins = assetManager.open(DIC_PATH);
        BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
      int index = 0;
      while (reader.ready()) {
        String key = reader.readLine();
        dic.put(key, index++);
      }
    }
  }

  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   * logits.
   */
  //Added a New Parameter Model
  @WorkerThread
  public synchronized List<QaAnswer> predict(String query, String content,
                                             String runtime,String Model,StringBuilder execStatus) {
    Log.v(TAG, "Convert Feature...");
    Feature feature = featureConverter.convert(query, content);
    //Toast.makeText(context, "Convert Feature Inside QA Client",Toast.LENGTH_SHORT).show();
    Log.v(TAG, "Set inputs...");
    float[][] inputIds = new float[1][MAX_SEQ_LEN];
    int[][] inpIds = new int[1][MAX_SEQ_LEN];
    float[][] inputMask = new float[1][MAX_SEQ_LEN];
    float[][] segmentIds = new float[1][MAX_SEQ_LEN];
    float[][] startLogits = new float[1][MAX_SEQ_LEN];
    float[][] endLogits = new float[1][MAX_SEQ_LEN];

    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      inputIds[0][j] = feature.inputIds[j];
      inpIds[0][j] = feature.inputIds[j];
      inputMask[0][j] = feature.inputMask[j];
      segmentIds[0][j] = feature.segmentIds[j];
    }

//    Object[] inputs = {inputIds, inputMask, segmentIds};
    Map<Integer, Object> output = new HashMap<>();
    output.put(0, startLogits);
    output.put(1, endLogits);

    Log.v(TAG, "Run inference...");
    if (runtime.equals("DSP")) {
      Log.i(TAG, "Sending Inf request to SNPE DSP");

      long htpSTime = System.currentTimeMillis();
      String dsp_logs = inferSNPE(runtime,Model,inputIds[0], inputMask[0], segmentIds[0],
              MAX_SEQ_LEN, startLogits[0], endLogits[0]);
      long htpETime = System.currentTimeMillis();
      long htpTime = htpETime - htpSTime;
      Log.i(TAG, "DSP Exec took : " + htpTime + "ms");

      if (! dsp_logs.isEmpty()) {
        Log.i(TAG, "DSP Exec status : " + dsp_logs);
        execStatus.append(dsp_logs);
      }
//      Log.i(TAG, "DSP: Startlogits = " + Arrays.toString(startLogits[0]));
    } else {
      Log.i(TAG, "Sending Inf request to SNPE CPU");
      String cpu_logs = inferSNPE(runtime,Model,inputIds[0], inputMask[0], segmentIds[0],
              MAX_SEQ_LEN, startLogits[0], endLogits[0]);

      if (! cpu_logs.isEmpty()) {
        Log.i(TAG, "CPU Exec status : " + cpu_logs);
        execStatus.append(cpu_logs);
      }
//      Log.i(TAG, "predict: Startlogits = " + Arrays.toString(startLogits[0]));
    }

    Log.v(TAG, "Convert logits to answers...");
    List<QaAnswer> answers = getBestAnswers(startLogits[0], endLogits[0], feature);
    Log.v(TAG, "Finish.");
    return answers;
  }

  /** Find the Best N answers & logits from the logits array and input feature. */
  private synchronized List<QaAnswer> getBestAnswers(
      float[] startLogits, float[] endLogits, Feature feature) {
    // Model uses the closed interval [start, end] for indices.
    int[] startIndexes = getBestIndex(startLogits, feature.tokenToOrigMap);
    int[] endIndexes = getBestIndex(endLogits, feature.tokenToOrigMap);

    List<QaAnswer.Pos> origResults = new ArrayList<>();
    for (int start : startIndexes) {
      for (int end : endIndexes) {
        if (end < start) {
          continue;
        }
        int length = end - start + 1;
        if (length > MAX_ANS_LEN) {
          continue;
        }
        origResults.add(new QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]));
      }
    }

    Collections.sort(origResults);

    List<QaAnswer> answers = new ArrayList<>();
    for (int i = 0; i < origResults.size(); i++) {
      if (i >= PREDICT_ANS_NUM) {
        break;
      }

      String convertedText;
      if (origResults.get(i).start > 0) {
        convertedText = convertBack(feature, origResults.get(i).start, origResults.get(i).end);
      } else {
        convertedText = "";
      }
      QaAnswer ans = new QaAnswer(convertedText, origResults.get(i));
      answers.add(ans);
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  @WorkerThread
  private synchronized int[] getBestIndex(float[] logits, Map<Integer, Integer> tokenToOrigMap) {
    List<QaAnswer.Pos> tmpList = new ArrayList<>();
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
      if (tokenToOrigMap.containsKey(i + OUTPUT_OFFSET)) {
        tmpList.add(new QaAnswer.Pos(i, i, logits[i]));
      }
    }

    Collections.sort(tmpList);

    int[] indexes = new int[PREDICT_ANS_NUM];
    for (int i = 0; i < PREDICT_ANS_NUM; i++) {
      indexes[i] = tmpList.get(i).start;
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  @WorkerThread
  private static String convertBack(Feature feature, int start, int end) {
     // Shifted index is: index of logits + offset.
    int shiftedStart = start + OUTPUT_OFFSET;
    int shiftedEnd = end + OUTPUT_OFFSET;
    int startIndex = feature.tokenToOrigMap.get(shiftedStart);
    int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
    // end + 1 for the closed interval.
    String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
    return ans;
  }

  /**
   * A native method that is implemented by the 'qa' native library,
   * which is packaged with this application.
   */
  public native String queryRuntimes(String nativeDirPath);
  public native String initSNPE(AssetManager assetManager, String Model);
  public native String inferSNPE(String runtime,String Model, float[] input_ids,
                                 float[] attn_masks, float[] seg_ids,
                                 int arraySizes,
                                 float[] startLogits, float[] endLogits);
}
