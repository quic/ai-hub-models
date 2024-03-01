// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "StreamEncode.h"

int StreamEncode::gst_wayland_pipeline_init(string output_type) 
{
    gst_init(nullptr, nullptr);
    data.pipeline.reset(gst_pipeline_new("pipeline"), [](GstElement *elem)
                        {
                            gst_element_set_state(elem, GST_STATE_NULL);
                            gst_object_unref(elem);
                        });
                        
    auto DefaultUnRefGstElement = [](GstElement *elem) { 
        // Per GStreamer design pipeline parent manage
        // GstElement instead of unreffing the object directly
    };

    data.appsrc.reset(gst_element_factory_make("appsrc", "appsrc"), DefaultUnRefGstElement);
    data.vidconv.reset(gst_element_factory_make("videoconvert", "vidconv"), DefaultUnRefGstElement);
    data.videoscale.reset(gst_element_factory_make("videoscale", "videoscale"), DefaultUnRefGstElement);
    data.waylandsink.reset(gst_element_factory_make("waylandsink", "waylandsink"), DefaultUnRefGstElement);

    if (!data.pipeline.get() || !data.appsrc.get() || !data.vidconv.get() || !data.videoscale.get() || !data.waylandsink.get())
    {
        LOG_ERROR("[not all element created,(%s)(%s)(%s)(%s)(%s)]\n",
            !data.pipeline.get() ? "ng" : "ok",
            !data.appsrc.get() ? "ng" : "ok",
            !data.vidconv.get() ? "ng" : "ok",
            !data.videoscale.get() ? "ng" : "ok",
            !data.waylandsink.get() ? "ng" : "ok");
        return QS_ERROR;
    }
    
    gst_bin_add_many(GST_BIN(data.pipeline.get()),
                        data.appsrc.get(),
                        data.vidconv.get(),
                        data.videoscale.get(),
                        data.waylandsink.get(),
                        NULL);

    GstCaps *caps = gst_caps_from_string("video/x-raw, framerate=30/1,width=1280, height=720,format=BGR");
    g_object_set(data.appsrc.get(), "caps", caps, NULL);
    gst_caps_unref(caps);

    g_object_set(G_OBJECT(data.waylandsink.get()), "async", true, NULL);
    g_object_set(G_OBJECT(data.waylandsink.get()), "sync", false, NULL);
    g_object_set (G_OBJECT (data.waylandsink.get()), "fullscreen", true, NULL);


    gst_element_sync_state_with_parent(data.waylandsink.get());


    if (!gst_element_link(data.appsrc.get(), data.vidconv.get()))
    {
        LOG_ERROR("Link Fail %s %s \n", GST_ELEMENT_NAME(data.appsrc.get()), GST_ELEMENT_NAME(data.vidconv.get()));
        return QS_ERROR;
    }

    if (!gst_element_link(data.vidconv.get(), data.videoscale.get()))
    {
        LOG_ERROR("Link Fail %s %s \n", GST_ELEMENT_NAME(data.vidconv.get()), GST_ELEMENT_NAME(data.videoscale.get()));
        return QS_ERROR;
    }
    if (!gst_element_link(data.videoscale.get(), data.waylandsink.get()))
    {
        LOG_ERROR("Link Fail %s %s \n", GST_ELEMENT_NAME(data.videoconvert.get()), GST_ELEMENT_NAME(data.waylandsink.get()));
        return QS_ERROR;
    }

    return QS_SUCCESS;
}


int StreamEncode::Initialization(string output_type)
{
    if(0 == output_type.compare("wayland"))
     {
        return gst_wayland_pipeline_init(output_type);
    }
    else 
    {
        LOG_ERROR("Stream Type does not configured");
        return QS_ERROR;
    }

    return QS_SUCCESS;

}

int StreamEncode::Loop()
{
        /* Start playing */
        GstStateChangeReturn ret = gst_element_set_state(data.pipeline.get(), GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            LOG_ERROR("Unable to set the pipeline to the playing state.\n");
            return QS_ERROR;
        }

        /* Listen to the bus */
        bus.reset(gst_element_get_bus(data.pipeline.get()), [](GstBus *obj)
                  { gst_object_unref(obj); });

        GstMessageType msgType;
        gchar *debug_info;
        GError *err;
        GstMessage *msg = nullptr;
        do
        {
            msgType = static_cast<GstMessageType>(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS);
            msg = gst_bus_timed_pop_filtered(bus.get(), GST_CLOCK_TIME_NONE, msgType);

            /* Parse message */
            if (msg != NULL)
            {
                switch (GST_MESSAGE_TYPE(msg))
                {
                case GST_MESSAGE_ERROR:
                    gst_message_parse_error(msg, &err, &debug_info);
                    LOG_ERROR("Error received from element %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
                    g_clear_error(&err);
                    g_free(debug_info);
                    terminate = TRUE;
                    break;
                case GST_MESSAGE_EOS:
                    LOG_INFO("End-Of-Stream reached. Encoder\n");
                    terminate = TRUE;
                    break;
                case GST_MESSAGE_STATE_CHANGED:
                    /* We are only interested in state-changed messages from the pipeline */
                    if (GST_MESSAGE_SRC(msg) == GST_OBJECT(data.pipeline.get()))
                    {
                        GstState old_state, new_state, pending_state;
                        gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                    }
                    break;
                default:
                    /* We should not reach here */
                    LOG_ERROR("Unexpected message received.\n");
                    break;
                }
                gst_message_unref(msg);
            }
        } while (!terminate);
        /* Free resources */
        gst_element_set_state(data.pipeline.get(), GST_STATE_NULL);
        return QS_SUCCESS;
}

void StreamEncode::UnInitialization()
{
        LOG_DEBUG("UnInitialization \n");
}

void StreamEncode::PushData(uint8_t *pushData, int len)
{
        GstBuffer *buffer = gst_buffer_new_and_alloc(len);
        gst_buffer_fill(buffer, 0, pushData, len);
        static GstClockTime timestamp = 0;
        GST_BUFFER_PTS(buffer) = timestamp;
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 30);

        timestamp += GST_BUFFER_DURATION(buffer);
        GstFlowReturn ret = GST_FLOW_OK;

        g_signal_emit_by_name(GST_APP_SRC(data.appsrc.get()), "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);

        if ((ret != GST_FLOW_OK))
        {
            LOG_ERROR("Error with gst_app_src_push_buffer for view_pipeline, return = %d \n", ret);
        }
}

void StreamEncode::Stop()
{
    terminate = TRUE;
    gst_app_src_end_of_stream(GST_APP_SRC(data.appsrc.get())); // send  eos
}

static void EncodeThreadFunc(shared_ptr<StreamEncode> encodePtr)
{
    int ret;
    ret = encodePtr->Loop();
    if(ret == QS_ERROR)
        LOG_ERROR("Failed to run the gstreamer pipeline\n");

}

void EncodeController::CreateEncoder(std::shared_ptr<SolutionConfiguration> sol_conf)
{
    int streamId = sol_conf->input_config->StreamNumber;
    string outputType = sol_conf->output_type;
    shared_ptr<StreamEncode> encodePtr = make_shared<StreamEncode>();
    encodePtr->Initialization(outputType);
    encoders.insert(pair<int, shared_ptr<StreamEncode>>(streamId, encodePtr));

    std::thread encodeThread = std::thread(EncodeThreadFunc, encodePtr);
    threads.emplace_back(move(encodeThread));
}

void EncodeController::EncodeFrame(int streamId, uint8_t *pushData, int len)
{
    encoders[streamId]->PushData(pushData, len);
}

void EncodeController::EndOfStream(int streamId)
{
    encoders[streamId]->Stop();
}

void EncodeController::InterruptClose()
{
        map<int, shared_ptr<StreamEncode>>::reverse_iterator iter;

        for (iter = encoders.rbegin(); iter != encoders.rend(); iter++)
        {
            iter->second->Stop();
        }
        
        for (size_t i = 0; i < threads.size(); i++)
        {
            threads[i].join();
        }
}

void EncodeController::Stop()
{
    map<int, shared_ptr<StreamEncode>>::reverse_iterator iter;

    for (iter = encoders.rbegin(); iter != encoders.rend(); iter++)
    {
        iter->second->Stop();
    }

    for (size_t i = 0; i < threads.size(); i++)
    {
        threads[i].join();
    }
}
