#ifndef STAGE_H_
#define STAGE_H_

#include <future>
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
                using size_type = typename input_side<InputT>::size_type;

            public:
                template <class... Args>
                stage(Args&&... args)
                : StageT(this->input_queue_, this, std::forward<Args>(args)...)
                , input_side<input_type>()
                , output_side<output_type>()
                {}

                template <class... Args>
                stage(size_type s, Args&&... args)
                : Implementation(std::forward<Args>(args)...)
                , input_side<input_type>(s)
                , output_side<output_type>()
                {}

                auto run() -> void
                {
                    StageT::run();
                }
        };
    }
}



#endif /* STAGE_H_ */
