#ifndef DDRF_PIPELINE_INPUT_SIDE_H_
#define DDRF_PIPELINE_INPUT_SIDE_H_

#include <type_traits>
#include <utility>

#include <boost/lockfree/spsc_queue.hpp>

namespace ddrf
{
    namespace pipeline
    {
        template <class InputT>
        class input_side
        {
            public:
                using queue_type = boost::lockfree::spsc_queue<InputT>;
                using size_type = typename queue_type::size_type;

            public:
                input_side() = default;
                input_side(size_type s) : input_queue_{s} {}

                template <class T>
                auto input(T&& t) -> typename std::enable_if<std::is_same<InputT, T>::value, void>::type
                {
                    input_queue_.push(std::forward<T>(t));
                }

            private:
                queue_type input_queue_;
        };
    }
}



#endif /* DDRF_PIPELINE_INPUT_SIDE_H_ */
