#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// A fixed-size worker pool used to record secondary command buffers in parallel. dispatch()
// runs a batch of tasks across the workers and blocks until every task has finished, which is
// exactly the fork/join shape a per-frame parallel recording step needs.
class ThreadPool {
public:
    explicit ThreadPool(unsigned threadCount);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void dispatch(std::vector<std::function<void()>> tasks);
    unsigned size() const { return static_cast<unsigned>(workers_.size()); }

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable taskReady_;
    std::condition_variable batchDone_;
    size_t pending_ = 0; // tasks submitted but not yet completed
    bool stop_ = false;
};
