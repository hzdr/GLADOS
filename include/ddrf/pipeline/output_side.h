#ifndef DDRF_PIPELINE_OUTPUT_SIDE_H_
#define DDRF_PIPELINE_OUTPUT_SIDE_H_

#include <type_traits>
#include <utility>

#include <ddrf/pipeline/input_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class OutputT>
        class output_side
        {
            public:
                template <class T>
                auto output(T&& t)
                -> typename std::enable_if<std::is_same<T, OutputT>::value, void>::type
                {
                    if(next_ == nullptr)
                        return;

                    next_->input(std::forward<T>(t));
                }

                auto attach(input_side<OutputT>* next) noexcept
                -> void
                {
                    next_ = next;
                }

            private:
                input_side<OutputT>* next_;
        };

        template <>
        class output_side<void>
        {
        };
    }
}



#endif /* DDRF_PIPELINE_OUTPUT_SIDE_H_ */
