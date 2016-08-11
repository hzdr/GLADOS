#ifndef SOURCE_STAGE_H_
#define SOURCE_STAGE_H_

#include <utility>

#include <ddrf/pipeline/output_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class SourceT>
        class source_stage : public SourceT
                           , public output_side<typename SourceT::output_type>
        {
            public:
                using output_type = typename SourceT::output_type;

            public:
                template <class... Args>
                source_stage(Args&&... args)
                : SourceT(this, std::forward<Args>(args)...), output_side<output_type>()
                {}

                auto run() -> void
                {
                    SourceT::run();
                }
        };
    }
}

#endif /* SOURCE_STAGE_H_ */
