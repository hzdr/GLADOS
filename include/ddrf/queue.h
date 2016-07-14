#ifndef DDRF_QUEUE_H_
#define DDRF_QUEUE_H_

#include <atomic>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

namespace ddrf
{
    template <class T>
    class queue
    {
        public:
            queue() noexcept = default;
            ~queue() = default;

            template <class Item>
            auto push(Item&& i) -> typename std::enable_if<std::is_same<T, Item>::value, void>::type
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                queue_.push(std::forward<Item>(i));

                lock_.clear(std::memory_order_release);
            }

            auto pop() -> T
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                auto ret = std::move(queue_.front());
                queue_.pop();

                lock_.clear(std::memory_order_release);
            }

        private:
            std::atomic_flag lock_;
            std::queue<T> queue_;
    };
}



#endif /* DDRF_QUEUE_H_ */
