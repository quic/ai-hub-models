// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#ifndef DECODE_QUEUE_H
#define DECODE_QUEUE_H

#include "Detection.h"
#include <condition_variable>
#include <list>
#include <locale>
#include <mutex>

static const int DEFAULT_MAX_QUEUE_SIZE = 64;

class DecodeQueue
{
public:
    DecodeQueue(uint32_t maxSize = DEFAULT_MAX_QUEUE_SIZE) : max_size_(maxSize), is_stoped_(false) {}
    ~DecodeQueue() {}
    int Dequeue(shared_ptr<DetectionItem>& item, unsigned int timeOutMs);
    int Enqueue(const shared_ptr<DetectionItem>& item, bool isWait);
    void Unlock();
    std::list<shared_ptr<DetectionItem>> GetRemainItems();
    int IsEmpty();
private:
    std::list<shared_ptr<DetectionItem>> queue_;
    std::mutex mutex_;
    std::condition_variable empty_cond_;
    std::condition_variable full_cond_;
    uint32_t max_size_;
    bool is_stoped_;
};

#endif