#ifndef DDRF_PIPELINE_PORT_H_
#define DDRF_PIPELINE_PORT_H_

#include <type_traits>
#include <utility>

#include <ddrf/pipeline/input_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class OutputT, class OutputM>
        class port
        {
            public:
                template <class T>
                auto forward(T&& t) -> typename std::enable_if<std::is_same<OutputT, T>::value, void>::type
                {
                    next_->input(std::forward<T>(t));
                }

                template <class M>
                auto forward(M&& m) -> typename std::enable_if<std::is_same<OutputM, M>::value, void>::type
                {
                    next_->input(std::forward<M>(m));
                }

                auto attach(input_side<OutputT, OutputM>* next) -> void
                {
                    next_ = next;
                }

            private:
                input_side<OutputT, OutputM>* next_;
        };
    }
}

#endif /* DDRF_PIPELINE_PORT_H_ */
