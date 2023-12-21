// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef INC_STREAM_DECODE_H
#define INC_STREAM_DECODE_H

#include "DecodeQueue.h"
#include "Configuration.h"
#include <iostream>
#include <string>
#include <memory>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <iostream>
#include <string>
#include <map>
#include <thread>

using namespace std;
/* Structure to contain all our information, so we can pass it to callbacks */
typedef struct _PipelineData
{
    shared_ptr<GstElement> pipeline;
    shared_ptr<GstElement> source;
    shared_ptr<GstElement> main_capsfilter;;
    shared_ptr<GstElement> videoDepay;
    shared_ptr<GstElement> videoParse;
    shared_ptr<GstElement> h264dec;
    shared_ptr<GstElement> transform;
    shared_ptr<GstElement> sink;
} PipelineData;

typedef struct _FrameProcessData
{
    uint32_t frameId;
    int interval = 25;
    shared_ptr<DecodeQueue> blockQueue;
    string streamName;
    int StreamId;
} FrameProcessData;
 

class StreamDecode
{
public:
    StreamDecode(std::string streamtype, std::string rtspUrl);
    ~StreamDecode();
    int Initialization(shared_ptr<DecodeQueue> &queue);
    void UnInitialization();
    void DecodeAndInference();
    void SetSkipFrame(int interval);
    void SetStreamName(string name);
    void SetStreamId(int uuid);

    static void OnPadAdd(GstElement *element, GstPad *pad, gpointer data);
    static GstFlowReturn OnAppsinkNewSample(GstElement *appsink, gpointer user_data);
    void Stop();
protected:
    static void UnRefElement(GstElement *elem);

private:
    PipelineData data_;
    shared_ptr<GstBus> bus_ = nullptr;
    bool terminate_ = FALSE;
    std::string StreamType;
    FrameProcessData *frameProcess_ = nullptr;
    int gst_camera_pipeline_init();
};

class CaptureController
{
    public:
        void CreateCapture(shared_ptr<InputConfiguration> &pipeline_config, shared_ptr<DecodeQueue> &gDecodeQueue);
        void EndOfStream(int streamId);
        void StopAll();
        void InterruptClose();

    private:
        map<int,shared_ptr<StreamDecode>> decoder;
        vector<std::thread> threads;
};

#endif