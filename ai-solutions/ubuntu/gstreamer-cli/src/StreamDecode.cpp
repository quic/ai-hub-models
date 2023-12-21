// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "StreamDecode.h"

GstFlowReturn StreamDecode::OnAppsinkNewSample(GstElement *appsink, gpointer user_data)
{
    FrameProcessData *frameProcess = reinterpret_cast<FrameProcessData *>(user_data);
    GstSample *sample = NULL;
    GstBuffer *buffer = NULL;
    GstMapInfo map;
    const GstStructure *info = NULL;
    GstCaps *caps = NULL;
    GstFlowReturn ret = GST_FLOW_OK;
    int sample_width = 0;
    int sample_height = 0;

    g_signal_emit_by_name(appsink, "pull-sample", &sample, &ret);
    if (ret != GST_FLOW_OK)
    {
        LOG_ERROR("can't pull GstSample.");
        return ret;
    }

    if (sample)
    {
        buffer = gst_sample_get_buffer(sample);
        if (buffer == NULL)
        {
            LOG_ERROR("get buffer is null");
            goto exit;
        }

        gst_buffer_map(buffer, &map, GST_MAP_READ);

        caps = gst_sample_get_caps(sample);
        if (caps == NULL)
        {
            LOG_ERROR("get caps is null");
            goto exit;
        }

        info = gst_caps_get_structure(caps, 0);
        if (info == NULL)
        {
            LOG_ERROR("get info is null");
            goto exit;
        }
       
        gst_structure_get_int(info, "width", &sample_width);
        gst_structure_get_int(info, "height", &sample_height);

        
        if (map.data == NULL)
        {
            LOG_ERROR("appsink buffer data empty");
            return GST_FLOW_OK;
        }

        frameProcess->frameId += 1;  
        LOG_DEBUG("Frame ID=%d \n", frameProcess->frameId);
        if (frameProcess->frameId % frameProcess->interval == 0)
        {
            shared_ptr<DetectionItem> detail(new DetectionItem());
            detail->Size = (uint32_t)map.size;
            detail->Width = sample_width;
            detail->Height = sample_height;
            detail->FrameId = frameProcess->frameId;
            detail->StreamName = frameProcess->streamName;
            detail->StreamId = frameProcess->StreamId;

            uint8_t *imgBuf = new uint8_t[map.size];
            memcpy(static_cast<uint8_t *>(imgBuf), map.data, map.size);
            detail->ImageBuffer.reset((uint8_t *)imgBuf, [](uint8_t *p)
                                      { delete[] (p); });

            int ret = frameProcess->blockQueue->Enqueue(detail,true);
            if (ret != 0)
            {
                LOG_ERROR("Enqueue Fail = %d \n", ret);
            }
        }
    }

exit:
    if (buffer)
    {
        gst_buffer_unmap(buffer, &map);
    }
    if (sample)
    {
        gst_sample_unref(sample);
    }
    return GST_FLOW_OK;
}

void StreamDecode::OnPadAdd(GstElement *element, GstPad *pad, gpointer data)
{
    // Link two Element with named pad
    GstPad *sink_pad = gst_element_get_static_pad(GST_ELEMENT(data), "sink");
    if (gst_pad_is_linked(sink_pad))
    {
        LOG_INFO("rtspsrc and depay are already linked. Ignoring\n");
        return;
    }
    gst_element_link_pads(element, gst_pad_get_name(pad), GST_ELEMENT(data), "sink");
}

StreamDecode::StreamDecode(std::string streamtype, std::string url)
{
    this->StreamType = streamtype;
    frameProcess_ = new FrameProcessData();
    frameProcess_->frameId = 1;
    this->frameProcess_->streamName = url;
}

StreamDecode::~StreamDecode()
{
    UnInitialization();
    if (frameProcess_ != nullptr)
    {
        delete frameProcess_;
        frameProcess_ = nullptr;
    }
}

void StreamDecode::UnInitialization()
{
    terminate_ = true;
}
void StreamDecode::DecodeAndInference()
{
    GstStateChangeReturn ret;
    shared_ptr<GstMessage> msg = nullptr;
    /* Start playing */
    ret = gst_element_set_state(data_.pipeline.get(), GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        LOG_ERROR("Unable to set the pipeline to the playing state.\n");
        return;
    }

    /* Listen to the bus */
    bus_.reset(gst_element_get_bus(data_.pipeline.get()), [](GstBus *obj)
               { gst_object_unref(obj); });

    GstMessageType msgType;
    do
    {
        msgType = static_cast<GstMessageType>(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
        msg.reset(gst_bus_timed_pop_filtered(bus_.get(), GST_CLOCK_TIME_NONE, msgType), [](GstMessage *m)
                  { gst_message_unref(m); });
        /* Parse message */
        if (msg != NULL)
        {
            GError *err;
            gchar *debug_info;
            
            switch (GST_MESSAGE_TYPE(msg.get()))
            {
            case GST_MESSAGE_ERROR:
                gst_message_parse_error(msg.get(), &err, &debug_info);
                LOG_ERROR("Error received from element = %s \t %s\n", GST_OBJECT_NAME(msg.get()->src), err->message);
                g_clear_error(&err);
                g_free(debug_info);
                terminate_ = true;
                break;
            case GST_MESSAGE_EOS:
                LOG_INFO("Stream = %s \t End-Of-Stream reached. total Frame = %d \n ", frameProcess_->streamName.c_str(), frameProcess_->frameId);
                terminate_ = true;
                break;
            case GST_MESSAGE_STATE_CHANGED:
                if (GST_MESSAGE_SRC(msg.get()) == GST_OBJECT(data_.pipeline.get()))
                {
                    GstState old_state, new_state, pending_state;
                    gst_message_parse_state_changed(msg.get(), &old_state, &new_state, &pending_state);
                }
                break;
            default:
                LOG_ERROR("Unexpected message received.\n");
                break;
            }
        }
    } while (!terminate_);

    gst_element_set_state(data_.pipeline.get(), GST_STATE_NULL);
}

void StreamDecode::UnRefElement(GstElement *elem)
{
    // LOG_DEBUG("Pipeline parent manage this object instead of unreffing the object directly: %s\n", elem->object.name);
}

int StreamDecode::gst_camera_pipeline_init() 
{
    GstCaps *filtercaps;
    /* Initialize GStreamer */
    gst_init(nullptr, nullptr);
    /* Create the empty pipeline */
    data_.pipeline.reset(gst_pipeline_new("decode-pipeline"), [](GstElement *elem) {
        gst_element_set_state(elem, GST_STATE_NULL);
        gst_object_unref(elem); });

    data_.source.reset(gst_element_factory_make("qtiqmmfsrc", "source"), UnRefElement);
    data_.main_capsfilter.reset(gst_element_factory_make ("capsfilter", "main_capsfilter"), UnRefElement);
    data_.transform.reset(gst_element_factory_make("qtivtransform", "transform"), UnRefElement);
    data_.sink.reset(gst_element_factory_make("appsink", "sink"), UnRefElement);

    if (!data_.pipeline.get() || !data_.source.get() || !data_.main_capsfilter.get() || !data_.transform.get() || !data_.sink.get())
    {
        LOG_ERROR("Not all elements could be created.");
        return QS_ERROR;
    }

    filtercaps = gst_caps_new_simple ("video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);

    gst_caps_set_features (filtercaps, 0,
        gst_caps_features_new ("memory:GBM", NULL));
    g_object_set (data_.main_capsfilter.get(), "caps", filtercaps, NULL);
    gst_caps_unref (filtercaps);

    gst_bin_add_many(GST_BIN(data_.pipeline.get()),  data_.source.get(), data_.main_capsfilter.get(), 
        data_.transform.get(), data_.sink.get(), NULL);
    if (!gst_element_link_many(data_.source.get(), data_.main_capsfilter.get(), data_.transform.get(), data_.sink.get(), NULL))
    {
        LOG_ERROR("Elements could not be linked.\n");
        gst_bin_remove_many (GST_BIN(data_.pipeline.get()),  data_.source.get(), data_.main_capsfilter.get(), 
            data_.transform.get(), data_.sink.get(), NULL);
        return QS_ERROR;
    }
    shared_ptr<GstCaps> caps(gst_caps_from_string("video/x-raw,format=BGR"), [](GstCaps *caps)
                            { gst_caps_unref(caps); });
    g_object_set(data_.sink.get(), "caps", caps.get(), NULL);

    /* Configure appsink */
    g_object_set(data_.sink.get(), "emit-signals", TRUE, NULL);
    g_signal_connect(data_.sink.get(), "new-sample", G_CALLBACK(StreamDecode::OnAppsinkNewSample), frameProcess_);

    /* Connect to the pad-added signal */
    g_signal_connect(data_.source.get(), "pad-added", G_CALLBACK(StreamDecode::OnPadAdd), data_.transform.get());

    return QS_SUCCESS;

}

int StreamDecode::Initialization(shared_ptr<DecodeQueue> &queue)
{
    frameProcess_->blockQueue = queue;

    if(0 == StreamType.compare("camera")) {
        return gst_camera_pipeline_init();
    }
    else {
        LOG_ERROR("Stream Type does not configured");
        return QS_ERROR;
    }
}

void StreamDecode::SetSkipFrame(int interval)
{
    if (interval < 1)
    {
        return;
    }
    frameProcess_->interval = interval;
}

void StreamDecode::SetStreamName(string name)
{
    LOG_INFO("Set stream name =%s \n", name.c_str());
    frameProcess_->streamName = name;
}

void StreamDecode::SetStreamId(int uuid)
{
    frameProcess_->StreamId = uuid;
}

void StreamDecode::Stop()
{
    terminate_ = true;
    gboolean res = gst_element_send_event(data_.pipeline.get(), gst_event_new_eos());
    if (!res)
    {
        LOG_ERROR("Error occurred! EOS signal cannot be sent!\n\r");
    }
}

static void CaptureThreadFunc(shared_ptr<StreamDecode> decodePtr)
{
    decodePtr->DecodeAndInference();
}

void CaptureController::CreateCapture(shared_ptr<InputConfiguration> &input_config, shared_ptr<DecodeQueue> &queue)
{
    shared_ptr<StreamDecode> decodePtr = make_shared<StreamDecode>(input_config->StreamType, input_config->Url);

    decodePtr->SetStreamId(input_config->StreamNumber);

    decodePtr->Initialization(queue);
    decodePtr->SetStreamName("stream_" + to_string(input_config->StreamNumber));

    decodePtr->SetSkipFrame(input_config->SkipFrame);
    
    std::thread decodeThread = std::thread(CaptureThreadFunc, decodePtr);
    threads.emplace_back(move(decodeThread));

    decoder.insert(pair<int, shared_ptr<StreamDecode>>(input_config->StreamNumber, decodePtr));
}

void CaptureController::InterruptClose()
{
    map<int, shared_ptr<StreamDecode>>::reverse_iterator iter;
    
    for (iter = decoder.rbegin(); iter != decoder.rend(); iter++)
    {
        iter->second->Stop();
    }
}

void CaptureController::StopAll()
{
    map<int, shared_ptr<StreamDecode>>::reverse_iterator iter;

    for (size_t i = 0; i < threads.size(); i++)
    { 
        threads[i].join();
    }

    for (iter = decoder.rbegin(); iter != decoder.rend(); iter++)
    {
        iter->second->Stop();
    }
}
