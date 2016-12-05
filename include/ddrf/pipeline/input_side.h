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
 * Date: 14 July 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_PIPELINE_INPUT_SIDE_H_
#define GLADOS_PIPELINE_INPUT_SIDE_H_

#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

namespace glados
{
    namespace pipeline
    {
        template <class InputT>
        class input_side
        {
            private:
                using mutex_type = std::mutex;
                using write_lock = std::unique_lock<mutex_type>;

            public:
                using queue_type = std::queue<InputT>;
                using size_type = typename queue_type::size_type;

            public:
                input_side() : queue_{}, limit_{0} {};
                input_side(size_type limit) : queue_{}, limit_{limit} {}
                
                input_side(const input_side& other) = delete;
                auto operator=(const input_side& other) -> input_side& = delete;

                input_side(input_side&& other)
                {
                    auto&& lock = write_lock{other.mutex_};
                    queue_ = std::move(other.queue_);
                    limit_ = std::move(other.limit_);
                }

                auto operator=(input_side&& other) -> input_side&
                {
                    if(this != &other)
                    {
                        // prevent possible deadlock
                        auto&& this_lock = write_lock{mutex_, std::defer_lock};
                        auto&& other_lock = write_lock{other.mutex_, std::defer_lock};
                        std::lock(this_lock, other_lock);
                        queue_ = std::move(other.queue_);
                        limit_ = std::move(other.limit_);
                    }

                    return *this;
                }

                template <class T>
                auto input(T&& t) -> typename std::enable_if<std::is_same<InputT, T>::value, void>::type
                {
                    if(limit_ != 0)
                    {
                        while(queue_.size() >= limit_)
                            std::this_thread::yield();
                    }

                    auto&& lock = write_lock{mutex_};
                    queue_.push(std::forward<T>(t));
                }

                auto take() -> InputT
                {
                    while(queue_.empty())
                        std::this_thread::yield();

                    auto&& lock = write_lock{mutex_};

                    auto ret = std::move(queue_.front());
                    queue_.pop();

                    return ret;
                }

            private:
                queue_type queue_;
                size_type limit_;
                mutable mutex_type mutex_;
        };

        template <>
        class input_side<void>
        {
        };
    }
}

#endif /* GLADOS_PIPELINE_INPUT_SIDE_H_ */
