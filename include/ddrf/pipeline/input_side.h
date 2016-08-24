#ifndef DDRF_PIPELINE_INPUT_SIDE_H_
#define DDRF_PIPELINE_INPUT_SIDE_H_

#include <atomic>
#include <cstddef>
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

                    while(lock_.test_and_set(std::memory_order_acquire))
                        std::this_thread::yield();

                    queue_.push(std::forward<T>(t));
                    lock_.clear(std::memory_order_release);
                }

                auto take() -> InputT
                {
                    while(queue_.empty())
                        std::this_thread::yield();

                    while(lock_.test_and_set(std::memory_order_acquire))
                        std::this_thread::yield();

                    auto ret = std::move(queue_.front());
                    queue_.pop();

                    lock_.clear(std::memory_order_release);
                    return ret;
                }

            private:
                queue_type queue_;
                size_type limit_;
                std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
        };
    }
}



#endif /* DDRF_PIPELINE_INPUT_SIDE_H_ */
