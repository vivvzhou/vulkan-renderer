#include "core/ThreadPool.hpp"

ThreadPool::ThreadPool(unsigned threadCount) {
    if (threadCount == 0) {
        threadCount = 1;
    }
    workers_.reserve(threadCount);
    for (unsigned i = 0; i < threadCount; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    taskReady_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::dispatch(std::vector<std::function<void()>> tasks) {
    if (tasks.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ += tasks.size();
        for (auto& task : tasks) {
            queue_.push(std::move(task));
        }
    }
    taskReady_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    batchDone_.wait(lock, [this] { return pending_ == 0; });
}

void ThreadPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            taskReady_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop();
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--pending_ == 0) {
                batchDone_.notify_all();
            }
        }
    }
}
