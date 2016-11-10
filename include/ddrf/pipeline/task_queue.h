#ifndef DDRF_PIPELINE_TASK_QUEUE_H_
#define DDRF_PIPELINE_TASK_QUEUE_H_

#include <mutex>
#include <queue>
#include <utility>

namespace ddrf
{
    namespace pipeline
    {
        template <class TaskT>
        class task_queue
        {
                task_queue(const std::queue<TaskT>& queue)
                : queue_{queue}
                {}

                auto push(TaskT&& t) -> void
                {
                    auto&& lock = std::lock_guard<std::mutex>{mutex_};
                    queue_.push(std::move(t));
                }

                auto pop() -> TaskT
                {
                    auto&& lock = std::lock_guard<std::mutex>{mutex_};

                    auto ret = std::move(queue_.front());
                    queue_.pop();

                    return ret;
                }

                auto empty() const -> bool
                {
                    return queue_.empty();
                }

            private:
                std::queue<TaskT> queue_;
                std::mutex mutex_;
        };
    }
}



#endif /* DDRF_PIPELINE_TASK_QUEUE_H_ */
