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
 * Date: 11 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_PIPELINE_PIPELINE_H_
#define GLADOS_PIPELINE_PIPELINE_H_

#include <functional>
#include <future>
#include <type_traits>
#include <utility>
#include <vector>

#include <glados/pipeline/input_side.h>
#include <glados/pipeline/output_side.h>
#include <glados/pipeline/stage.h>
#include <glados/pipeline/task_queue.h>

namespace glados
{
    namespace pipeline
    {
        class pipeline_base
        {
            public:
                template <class Last>
                auto connect(Last& l) const noexcept
                -> typename std::enable_if<std::is_base_of<input_side<typename Last::input_type>, Last>::value, void>::type
                {}

                template <class First, class Second>
                auto connect(First& f, Second& s) const noexcept
                -> typename std::enable_if<std::is_base_of<output_side<typename First::output_type>, First>::value &&
                                           std::is_base_of<input_side<typename Second::input_type>, Second>::value, void>::type
                {
                    f.attach(&s);
                }

                template <class First, class Second, class... Rest>
                auto connect(First& f, Second& s, Rest&... rs) const noexcept -> void
                {
                    connect(f, s);
                    connect(s, rs...);
                }

                template <class StageT, class... Args>
                auto make_stage(Args&&... args) const -> stage<StageT>
                {
                    return stage<StageT>{std::forward<Args>(args)...};
                }
        };

        class pipeline : public pipeline_base
        {
            public:
                template <class Runnable>
                auto run(Runnable& r) -> void
                {

                    futures_.emplace_back(std::async(std::launch::async, &Runnable::run, &r));
                }

                template <class Runnable, class... Runnables>
                auto run(Runnable&& r, Runnables&&... rs) -> void
                {
                    run(std::forward<Runnable>(r));
                    run(std::forward<Runnables>(rs)...);
                }

                auto wait() -> void
                {
                    try
                    {
                        for(auto&& f : futures_)
                            f.get();
                    }
                    catch(...)
                    {
                        throw;
                    }
                }

            private:
                std::vector<std::future<void>> futures_;
        };

        template <class TaskT>
        class task_pipeline : public pipeline_base
        {
            public:
                task_pipeline(task_queue<TaskT>* queue) noexcept
                : queue_{queue}
                {}

                auto run() -> void
                {
                    exec_future_ = std::async(std::launch::async, &task_pipeline::internal_run, this);
                }

                template <class Runnable, class... Runnables>
                auto run(Runnable&& r, Runnables&&... rs) -> void
                {
                    store_funcs(std::forward<Runnable>(r));
                    run(std::forward<Runnables>(rs)...);
                }

                auto wait() -> void
                {
                    try
                    {
                        exec_future_.get();
                    }
                    catch(...)
                    {
                        throw;
                    }
                }

            private:
                template <class Runnable>
                auto store_funcs(Runnable& r) -> void
                {
                    auto run_func = std::bind(&Runnable::run, &r);
                    auto assign_func = std::bind(&Runnable::assign_task, &r, std::placeholders::_1);
                            
                    runs_.push_back(run_func);
                    assigns_.push_back(assign_func);
                }

                auto internal_run() -> void
                {
                    try
                    {
                        if(queue_ != nullptr)
                        {
                            while(!queue_->empty())
                            {
                                auto task = queue_->pop();

                                for(auto&& assign_func : assigns_)
                                    assign_func(task);

                                for(auto&& run_func : runs_)
                                    stage_futures_.emplace_back(std::async(std::launch::async, run_func));

                                for(auto&& f : stage_futures_)
                                    f.get();

                                stage_futures_.clear();
                            }
                        }
                    }
                    catch(...)
                    {
                        throw;
                    }
                }

            private:
                task_queue<TaskT>* queue_;
                std::vector<std::future<void>> stage_futures_;
                std::future<void> exec_future_;

                std::vector<std::function<void(TaskT)>> assigns_;
                std::vector<std::function<void()>> runs_;
        };
    }
}

#endif /* GLADOS_PIPELINE_PIPELINE_H_ */
