#ifndef DDRF_PIPELINE_SINK_STAGE_H_
#define DDRF_PIPELINE_SINK_STAGE_H_

#include <functional>
#include <utility>

#include <ddrf/pipeline/input_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class SinkT>
        class sink_stage : public SinkT
                         , public input_side<typename SinkT::input_type>
        {
            public:
                using input_type = typename SinkT::input_type;
                using size_type = typename input_side<input_type>::size_type;

            public:
                template <class... Args>
                sink_stage(Args&&... args)
                : SinkT(std::forward<Args>(args)...), input_side<input_type>()
                {}

                template <class... Args>
                sink_stage(size_type input_limit, Args&&... args)
                : SinkT(std::forward<Args>(args)...), input_side<input_type>(input_limit)
                {}

                auto run() -> void
                {
                    SinkT::set_input_function(std::bind(&input_side<input_type>::take, this));
                    SinkT::run();
                }
        };
    }
}



#endif /* SINK_STAGE_H_ */
