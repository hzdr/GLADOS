#ifndef STAGE_H_
#define STAGE_H_

#include <functional>
#include <utility>

#include <ddrf/pipeline/input_side.h>
#include <ddrf/pipeline/output_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class StageT>
        class stage : public StageT
                    , public input_side<typename StageT::input_type>
                    , public output_side<typename StageT::output_type>
        {
            public:
                using input_type = typename StageT::input_type;
                using output_type = typename StageT::output_type;
                using size_type = typename input_side<input_type>::size_type;

            public:
                template <class... Args>
                stage(Args&&... args)
                : StageT(std::forward<Args>(args)...)
                , input_side<input_type>()
                , output_side<output_type>()
                {}

                template <class... Args>
                stage(size_type input_limit, Args&&... args)
                : Implementation(std::forward<Args>(args)...)
                , input_side<input_type>(input_limit)
                , output_side<output_type>()
                {}

                auto run() -> void
                {
                    StageT::set_input_function(std::bind(&input_side<input_type>::take, this));
                    StageT::set_output_function(std::bind(&output_side<output_type>::output, this, std::placeholders::_1));
                    StageT::run();
                }
        };
    }
}



#endif /* STAGE_H_ */
