/*
 * This file is part of the GLADOS library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * GLADOS is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GLADOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with GLADOS. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Date: 10 November 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_PIPELINE_TASK_QUEUE_H_
#define GLADOS_PIPELINE_TASK_QUEUE_H_

#include <mutex>
#include <queue>
#include <utility>

namespace glados
{
    namespace pipeline
    {
        template <class TaskT>
        class task_queue
        {
            public:
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

#endif /* GLADOS_PIPELINE_TASK_QUEUE_H_ */
