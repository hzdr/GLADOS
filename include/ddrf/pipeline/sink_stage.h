#ifndef SINK_STAGE_H_
#define SINK_STAGE_H_

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
                using size_type = typename input_side<InputT>::size_type;

            public:
                template <class... Args>
                sink_stage(Args&&... args)
                : SinkT(this->input_queue_, std::forward<Args>(args)...), input_side<input_type>()
                {}

                template <class... Args>
                sink_stage(size_type s, Args&&... args)
                : SinkT(std::forward<Args>(args)...), input_side<input_type>(s)
                {}

                auto run() -> void
                {
                    SinkT::run();
                }
        };
    }
}



#endif /* SINK_STAGE_H_ */
