#ifndef DDRF_PIPELINE_PIPELINE_H_
#define DDRF_PIPELINE_PIPELINE_H_

#include <future>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddrf/pipeline/input_side.h>
#include <ddrf/pipeline/output_side.h>
#include <ddrf/pipeline/stage.h>
#include <ddrf/pipeline/task_queue.h>

namespace ddrf
{
    namespace pipeline
    {
        class pipeline_base
        {
            public:
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
                    connect(rs...);
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
                auto run(Runnable&& r) -> void
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

                template <class... Runnables>
                auto run(Runnables... rs) -> void
                {
                    exec_future_ = std::async(std::launch::async, &task_pipeline<TaskT>::internal_run, this, std::forward<Runnables>(rs)...);
                }

                auto wait() -> void
                {
                    try
                    {
                        exec_future_.get()
                    }
                    catch(...)
                    {
                        throw;
                    }
                }

            private:
                template <class Runnable>
                auto launch(TaskT t, Runnable&& r) -> void
                {
                    r.assign_task(t);
                    stage_futures_.emplace_back(std::async(std::launch::async, &Runnable::run, &r));
                }

                template <class Runnable, class... Runnables>
                auto launch(TaskT t, Runnable&& r, Runnables&&... rs) -> void
                {
                    launch(t, std::forward<Runnable>(r));
                    launch(t, std::forward<Runnables>(rs)...);
                }

                template <class Runnable, class... Runnables>
                auto internal_run(Runnable&& r, Runnables&&... rs) -> void
                {
                    if(queue_ != nullptr)
                    {
                        while(!queue->empty())
                        {
                            auto task = queue->pop();
                            assign_task(task, std::forward<Runnable>(r), std::forward<Runnable>(rs)...);

                            launch(std::forward<Runnable>(r), std::forward<Runnables>(rs)...);

                            for(auto&& f : stage_futures_)
                                f.get();

                            stage_futures_.clear();
                        }
                    }
                }

            private:
                task_queue<TaskT>* queue_;
                std::vector<std::future<void>> stage_futures_;
                std::future<void> exec_future_;
        };
    }
}



#endif /* PIPELINE_H_ */
