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

package com.qualcomm.qti.qa.ui;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.text.Editable;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.TextWatcher;
import android.text.method.ScrollingMovementMethod;
import android.text.style.BackgroundColorSpan;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.textfield.TextInputEditText;
import java.util.List;
import java.util.Locale;
import com.qualcomm.qti.R;
import com.qualcomm.qti.qa.ml.LoadDatasetClient;
import com.qualcomm.qti.qa.ml.QaAnswer;
import com.qualcomm.qti.qa.ml.QaClient;

/** Activity for doing Q&A on a specific dataset */
public class QaActivity extends AppCompatActivity {

  private static final String DATASET_POSITION_KEY = "DATASET_POSITION";
  private static final String TAG = "SNPE_Activity";
  private static final boolean DISPLAY_RUNNING_TIME = true;

  private TextInputEditText questionEditText;
  private TextView contentTextView;
  private TextToSpeech textToSpeech;

  private boolean questionAnswered = false;
  private String content;
  private Handler handler;
  private QaClient qaClient;
  final String[] model = {"alberta"};

  public static Intent newInstance(Context context, int datasetPosition) {
    Intent intent = new Intent(context, QaActivity.class);
    intent.putExtra(DATASET_POSITION_KEY, datasetPosition);
    return intent;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    Log.v(TAG, "onCreate");
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_qa);

    // Get content of the selected dataset.
    int datasetPosition = getIntent().getIntExtra(DATASET_POSITION_KEY, -1);
    LoadDatasetClient datasetClient = new LoadDatasetClient(this);

    // Show the dataset title.
    TextView titleText = findViewById(R.id.title_text);
    titleText.setText(datasetClient.getTitles()[datasetPosition]);

    // Show the text content of the selected dataset.
    content = datasetClient.getContent(datasetPosition);
    contentTextView = findViewById(R.id.content_text);
    contentTextView.setText(content);
    contentTextView.setMovementMethod(new ScrollingMovementMethod());

    // Setup question suggestion list.
    RecyclerView questionSuggestionsView = findViewById(R.id.suggestion_list);
    QuestionAdapter adapter =
        new QuestionAdapter(this, datasetClient.getQuestions(datasetPosition));
    adapter.setOnQuestionSelectListener(question -> answerQuestion(question));
    questionSuggestionsView.setAdapter(adapter);
    LinearLayoutManager layoutManager =
        new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false);
    questionSuggestionsView.setLayoutManager(layoutManager);


      //=========================== Model Selection ==============================//
      Spinner model_dropdown = findViewById(R.id.model_spinner);
      String[] model_items = new String[]{"alberta","electra_small","mobile_bert"};

      ArrayAdapter<String> model_adapter = new ArrayAdapter<String>(QaActivity.this,
              android.R.layout.simple_spinner_item,model_items);

      model_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
      model_dropdown.setAdapter(model_adapter);
      model_dropdown.setOnItemSelectedListener(
              new AdapterView.OnItemSelectedListener() {
                  @Override
                  public void onItemSelected(AdapterView<?> arg0, View arg1,
                                             int arg2, long arg3) {
                      model[0] =model_dropdown.getSelectedItem().toString();
                      Toast.makeText(QaActivity.this,"Model selected: "+ model[0],Toast.LENGTH_SHORT).show();

                      //Initializing the Selected Model
                      handler.post(
                              () -> {
                                  String initLogs = qaClient.loadModel(model[0]);
                                  if(!initLogs.isEmpty()) {
                                      Snackbar initSnackbar =
                                              Snackbar.make(contentTextView, initLogs, Snackbar.LENGTH_SHORT);
                                      initSnackbar.show();
                                  }
                                  qaClient.loadDictionary();
                              });
                  }
                  @Override
                  public void onNothingSelected(AdapterView<?> arg0) {
                      // TODO Auto-generated method stub
                      model[0] = "alberta";
                  }
              });

      //=========================== model Selection End==============================//

      //=========================== Runtime Selection ==============================//

      Spinner dropdown = findViewById(R.id.runtime_spinner);
      String[] items = new String[]{"DSP", "CPU"};

      ArrayAdapter<String> ddadapter = new ArrayAdapter<String>(QaActivity.this,
              android.R.layout.simple_spinner_item,items);

      ddadapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
      dropdown.setAdapter(ddadapter);
      final String[] runtime = {"DSP"};
      dropdown.setOnItemSelectedListener(
              new AdapterView.OnItemSelectedListener() {
                  @Override
                  public void onItemSelected(AdapterView<?> arg0, View arg1,
                                             int arg2, long arg3) {
                      runtime[0] =dropdown.getSelectedItem().toString();
                      Log.i("SPINNER: Dropdown selected is  ", runtime[0]);
                      Toast.makeText(QaActivity.this,"Runtime selected: "+ runtime[0],Toast.LENGTH_SHORT).show();
                  }
                  @Override
                  public void onNothingSelected(AdapterView<?> arg0) {
                      // TODO Auto-generated method stub
                      runtime[0] = "DSP";
                  }
              });
      //=========================== Runtime Selection End==============================//
    // Setup ask button.
    ImageButton askButton = findViewById(R.id.ask_button);
    askButton.setOnClickListener(
        view -> answerQuestion(questionEditText.getText().toString()));

    // Setup text edit where users can input their question.
    questionEditText = findViewById(R.id.question_edit_text);
    questionEditText.setOnFocusChangeListener(
        (view, hasFocus) -> {
          // If we already answer current question, clear the question so that user can input a new
          // one.
          if (hasFocus && questionAnswered) {
            questionEditText.setText(null);
          }
        });
    questionEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}

          @Override
          public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
            // Only allow clicking Ask button if there is a question.
            boolean shouldAskButtonActive = !charSequence.toString().isEmpty();
            askButton.setClickable(shouldAskButtonActive);
            askButton.setImageResource(
                shouldAskButtonActive ? R.drawable.ic_ask_active : R.drawable.ic_ask_inactive);
          }

          @Override
          public void afterTextChanged(Editable editable) {}
        });
    questionEditText.setOnKeyListener(
        (v, keyCode, event) -> {
          if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_ENTER) {
            answerQuestion(questionEditText.getText().toString());
          }
          return false;
        });
    // Setup QA client to and background thread to run inference.
    HandlerThread handlerThread = new HandlerThread("QAClient");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    qaClient = new QaClient(this);
  }

  @Override
  protected void onStart() {
    Log.v(TAG, "onStart");
    super.onStart();

    //Here Loading the particular Model
      //Here In the Init Part I'll Create a if-else loop and load the selected Model
      //This Model Loading Part is also done in selection part also
    handler.post(
        () -> {
          String initLogs = qaClient.loadModel(model[0]);
          if(!initLogs.isEmpty()) {
              Snackbar initSnackbar =
                      Snackbar.make(contentTextView, initLogs, Snackbar.LENGTH_SHORT);
              initSnackbar.show();
          }
          qaClient.loadDictionary();
        });

    //This is to say the answer using speech
    textToSpeech =
        new TextToSpeech(
            this,
            status -> {
              if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
              } else {
                textToSpeech = null;
              }
            });
  }

  @Override
  protected void onStop() {
    Log.v(TAG, "onStop");
    super.onStop();
    handler.post(() -> qaClient.unload());

    if (textToSpeech != null) {
      textToSpeech.stop();
      textToSpeech.shutdown();
    }
  }

  private void answerQuestion(String question) {
    question = question.trim();
    if (question.isEmpty()) {
      questionEditText.setText(question);
      return;
    }

    // Append question mark '?' if not ended with '?'.
    // This aligns with question format that trains the model.
    if (!question.endsWith("?")) {
      question += '?';
    }
    final String questionToAsk = question;
    questionEditText.setText(questionToAsk);

    // Delete all pending tasks.
    handler.removeCallbacksAndMessages(null);

    // Hide keyboard and dismiss focus on text edit.
    InputMethodManager imm =
        (InputMethodManager) getSystemService(AppCompatActivity.INPUT_METHOD_SERVICE);
    imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
    View focusView = getCurrentFocus();
    if (focusView != null) {
      focusView.clearFocus();
    }

    // Reset content text view
    contentTextView.setText(content);

    questionAnswered = false;

    Snackbar runningSnackbar =
        Snackbar.make(contentTextView, "Looking up answer...", Snackbar.LENGTH_INDEFINITE);
    runningSnackbar.show();

    // Run TF Lite model to get the answer.
    handler.post(
        () -> {
            Spinner dropdown = findViewById(R.id.runtime_spinner);
            String runtime=dropdown.getSelectedItem().toString();
            Log.i("SPINNER: Dropdown selected is  ", runtime);
            Spinner model_dropdown=findViewById(R.id.model_spinner);
            String model= model_dropdown.getSelectedItem().toString();

            StringBuilder execStatus = new StringBuilder ();

          long beforeTime = System.currentTimeMillis();

          //To predict the answer with given question and content
          // Here Runtime and Model is also passed as parameter to say which model to run
          //Model is already loaded when selecting the dropdown menu for model
          final List<QaAnswer> answers = qaClient.predict(questionToAsk, content, runtime,model, execStatus);
          long afterTime = System.currentTimeMillis();
          double totalSeconds = (afterTime - beforeTime) / 1000.0;

          if (!answers.isEmpty()) {
            // Get the top answer
            QaAnswer topAnswer = answers.get(0);
            // Show the answer.
            runOnUiThread(
                () -> {
                  runningSnackbar.dismiss();
                  presentAnswer(topAnswer);

                  String displayMessage = runtime + " runtime took : ";
                  if (DISPLAY_RUNNING_TIME) {
                    displayMessage = String.format("%s %.3f sec.", displayMessage, totalSeconds);
                  }
                  if (! execStatus.toString().isEmpty())
                      Snackbar.make(contentTextView, execStatus.toString(), Snackbar.LENGTH_LONG).show();
                  else
                      Snackbar.make(contentTextView, displayMessage, Snackbar.LENGTH_LONG).show();

                  questionAnswered = true;
                });
          }
        });
  }

  private void presentAnswer(QaAnswer answer) {
    // Highlight answer.
    Spannable spanText = new SpannableString(content);
    int offset = content.indexOf(answer.text, 0);
    if (offset >= 0) {
      spanText.setSpan(
          new BackgroundColorSpan(getColor(R.color.secondaryColor)),
          offset,
          offset + answer.text.length(),
          Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
    }
    contentTextView.setText(spanText);

    // Use TTS to speak out the answer.
    if (textToSpeech != null) {
      textToSpeech.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text);
    }
  }
}
