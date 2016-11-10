#ifndef DDRF_PIPELINE_INPUT_SIDE_H_
#define DDRF_PIPELINE_INPUT_SIDE_H_

#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

namespace ddrf
{
    namespace pipeline
    {
        template <class InputT>
        class input_side
        {
            public:
                using queue_type = std::queue<InputT>;
                using size_type = typename queue_type::size_type;

            public:
                input_side() : queue_{}, limit_{0} {};
                input_side(size_type limit) : queue_{}, limit_{limit} {}

                template <class T>
                auto input(T&& t) -> typename std::enable_if<std::is_same<InputT, T>::value, void>::type
                {
                    if(limit_ != 0)
                    {
                        while(queue_.size() >= limit_)
                            std::this_thread::yield();
                    }

                    auto&& lock = std::unique_lock<std::mutex>{mutex_};
                    queue_.push(std::forward<T>(t));
                }

                auto take() -> InputT
                {
                    while(queue_.empty())
                        std::this_thread::yield();

                    auto&& lock = std::unique_lock<std::mutex>{mutex_};

                    auto ret = std::move(queue_.front());
                    queue_.pop();

                    return ret;
                }

            private:
                queue_type queue_;
                size_type limit_;
                std::mutex mutex_;
        };

        template <>
        class input_side<void>
        {
        };
    }
}



#endif /* DDRF_PIPELINE_INPUT_SIDE_H_ */
