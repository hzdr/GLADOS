#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <future>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddrf/pipeline/input_side.h>
#include <ddrf/pipeline/output_side.h>
#include <ddrf/pipeline/sink_stage.h>
#include <ddrf/pipeline/source_stage.h>
#include <ddrf/pipeline/stage.h>

namespace ddrf
{
    namespace pipeline
    {
        class pipeline
        {
            public:
                template <class First, class Second>
                auto connect(First&& f, Second&& s) const noexcept
                -> typename std::enable_if<std::is_base_of<output_side<typename First::output_type>, First>::value &&
                                           std::is_base_of<input_side<typename Second::input_type>, Second>::value, void>::type
                {
                    f.attach(&s);
                }

                template <class StageT, class... Args>
                auto make_stage(Args&&... args) const -> stage<StageT>
                {
                    return stage<StageT>{std::forward<Args>(args)...};
                }

                template <class SourceT, class... Args>
                auto make_stage(Args&&... args) const -> source_stage<SourceT>
                {
                    return source_stage<SourceT>{std::forward<Args>(args)...};
                }

                template <class SinkT, class... Args>
                auto make_stage(Args&&... args) const -> sink_stage<SinkT>
                {
                    return sink_stage<SinkT>{std::forward<Args>(args)...};
                }

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
    }
}



#endif /* PIPELINE_H_ */
