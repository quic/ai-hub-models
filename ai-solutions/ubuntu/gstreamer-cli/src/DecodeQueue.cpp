// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
#include "DecodeQueue.h"

/** @brief To access frames from source
 * @param item to store the frame
 * @param timeOutMs to wait for frame till timeout
 * @return 0 if success
*/

int DecodeQueue::Dequeue(shared_ptr<DetectionItem> &item, unsigned int timeOutMs)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto realTime = std::chrono::milliseconds(timeOutMs);

    while (queue_.empty() && !is_stoped_)
    {
        empty_cond_.wait_for(lock, realTime);
    }
    /**
     * To check if pipeline is stopped
    */
    if (is_stoped_)
    {
        return 1;
    }
    /**
     * To check if queue is emtpy
    */
    else if (queue_.empty())
    {
        return 2;
    }
    else
    {
        item = queue_.front();
        queue_.pop_front();
    }

    full_cond_.notify_one();

    return 0;
}

/** @brief To enqueue the frames to display or save
 * @param item to push into the queue 
 * @param isWait to wait for frame  till timeout
*/

int DecodeQueue::Enqueue(const shared_ptr<DetectionItem> &item, bool isWait)
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.size() >= max_size_ && isWait && !is_stoped_)
    {
        full_cond_.wait(lock);
    }
     /**
     * To check if pipeline is stopped
    */
    if (is_stoped_)
    {
        return 1;
    }
    /**
     * To check if queue_ size is greater than max size
    */
    else if (queue_.size() >= max_size_)
    {
        return 3;
    }
    queue_.push_back(item);
    empty_cond_.notify_one();
    return 0;
}

/** @brief To stop the pipeline 
*/

void DecodeQueue::Unlock()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        is_stoped_ = true;
    }

    full_cond_.notify_all();
    empty_cond_.notify_all();
}

/** @brief To inference the remaining items
*/
std::list<shared_ptr<DetectionItem>> DecodeQueue::GetRemainItems()
{
    std::unique_lock<std::mutex> lock(mutex_);
     /**
     * To check if pipeline is stopped
    */
    if (!is_stoped_)
    {
        return std::list<shared_ptr<DetectionItem>>();
    }

    return queue_;
}

/** @brief To check if queue is empty
*/
int DecodeQueue::IsEmpty()
{
    return queue_.empty();
}