#ifndef DDRF_PIPELINE_INPUT_SIDE_H_
#define DDRF_PIPELINE_INPUT_SIDE_H_

#include <type_traits>
#include <utility>

#include <ddrf/queue.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class InputT, class InputM>
        class input_side
        {
            public:
                input_side() noexcept = default;

                template <class T>
                auto input(T&& t) -> typename std::enable_if<std::is_same<InputT, T>::value, void>::type
                {
                    input_queue_.push(std::forward<T>(t));
                }

                template <class M>
                auto input(M&& m) -> typename std::enable_if<std::is_same<InputM, M>::value, void>::type
                {
                    meta_queue_.push(std::forward<M>(m));
                }

            private:
                ddrf::queue<InputT> input_queue_;
                ddrf::queue<InputM> meta_queue_;
        };
    }
}



#endif /* DDRF_PIPELINE_INPUT_SIDE_H_ */
