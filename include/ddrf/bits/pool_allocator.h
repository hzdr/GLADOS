#ifndef DDRF_BITS_POOL_ALLOCATOR_H_
#define DDRF_BITS_POOL_ALLOCATOR_H_

#include <atomic>
#include <forward_list>
#include <functional>
#include <thread>
#include <type_traits>
#include <utility>

#include <ddrf/bits/memory_layout.h>

namespace ddrf
{
    template <class T, memory_layout ml, class InternalAlloc, class = typename std::enable_if<(ml == InternalAlloc::mem_layout)>::type>
    class pool_allocator {};

    /* 1D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>
    {
        public:
            static constexpr auto mem_layout = InternalAlloc::mem_layout;
            static constexpr auto mem_location = InternalAlloc::mem_location;
            static constexpr auto alloc_needs_pitch = InternalAlloc::alloc_needs_pitch;

            using value_type = T;
            using pointer = typename InternalAlloc::pointer;
            using const_pointer = typename InternalAlloc::const_pointer;
            using size_type = typename InternalAlloc::size_type;
            using difference_type = typename InternalAlloc::difference_type;
            using propagate_on_container_copy_assignment = std::true_type;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;
            using is_always_equal = std::true_type;
            using smart_pointer = typename InternalAlloc::template smart_pointer<std::function<void(T*)>>;

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, mem_layout, InternalAlloc>;
            };

        public:
            pool_allocator() = default;

            pool_allocator(size_type limit) noexcept
            : alloc_{}, list_{}, n_{}, limit_{limit}, current_{0}
            {}

            pool_allocator(pool_allocator&& other) noexcept
            : alloc_{std::move(other.alloc_)}, list_{std::move(other.list_)}
            , n_{other.n_}, limit_{other.limit_}, current_{other.current_.load()}, moved_{other.moved_}
            {
                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;
            }

            auto operator=(pool_allocator&& other) noexcept -> pool_allocator&
            {
                alloc_ = std::move(other.alloc_);
                list_ = std::move(other.list_);
                n_ = other.n_;
                limit_ = other.limit_;
                current_ = other.current_.load();
                moved_ = other.moved_;

                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;

                return *this;
            }

            pool_allocator(const pool_allocator& other) noexcept = delete;
            auto operator=(const pool_allocator& other) noexcept -> pool_allocator& = delete;

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type n) -> pointer
            {
                if(moved_)
                    return pointer{nullptr};

                if(n_ == 0)
                    n_ = n;

                auto ret = static_cast<pointer>(nullptr);

                if(limit_ != 0)
                {
                    while(current_.load() >= limit_)
                        std::this_thread::yield();
                }

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                if(list_.empty())
                    ret = alloc_.allocate(n_);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                ++current_;
                lock_.clear(std::memory_order_release);

                return ret;
            }

            auto allocate_smart(size_type n) -> smart_pointer
            {
                return smart_pointer{allocate(n), [this](T* ptr){ this->deallocate(ptr); }};
            }

            auto deallocate(pointer p, size_type = 0) noexcept -> void
            {
                if(moved_ || (p == nullptr))
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                --current_;
                lock_.clear(std::memory_order_release);
            }

            auto release() noexcept -> void
            {
                if(moved_)
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, n_);

                n_ = 0;
                current_.store(0);

                lock_.clear(std::memory_order_release);
            }

        private:
            InternalAlloc alloc_;
            std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
            std::forward_list<pointer> list_;
            size_type n_;
            size_type limit_;
            std::atomic_size_t current_;
            bool moved_ = false;
    };

    /* 2D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>
    {
        public:
            static constexpr auto mem_layout = InternalAlloc::mem_layout;
            static constexpr auto mem_location = InternalAlloc::mem_location;
            static constexpr auto alloc_needs_pitch = InternalAlloc::alloc_needs_pitch;

            using value_type = T;
            using pointer = typename InternalAlloc::pointer;
            using const_pointer = typename InternalAlloc::const_pointer;
            using size_type = typename InternalAlloc::size_type;
            using difference_type = typename InternalAlloc::difference_type;
            using propagate_on_container_copy_assignment = std::true_type;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;
            using is_always_equal = std::true_type;
            using smart_pointer = typename InternalAlloc::template smart_pointer<std::function<void(T*)>>;

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, mem_layout, InternalAlloc>;
            };

        public:
            pool_allocator() = default;

            pool_allocator(size_type limit)
            : alloc_{}, list_{}, x_{0}, y_{0}, limit_{limit}, current_{0}
            {}

            pool_allocator(pool_allocator&& other) noexcept
            : alloc_{std::move(other.alloc_)}, list_{std::move(other.list_)}
            , x_{other.x_}, y_{other.y_}, limit_{other.limit_}, current_{other.current_.load()}, moved_{other.moved_}
            {
                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;
            }

            auto operator=(pool_allocator&& other) noexcept -> pool_allocator&
            {
                alloc_ = std::move(other.alloc_);
                list_ = std::move(other.list_);
                x_ = other.x_;
                y_ = other.y_;
                limit_ = other.limit_;
                current_ = other.current_.load();
                moved_ = other.moved_;
                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;

                return *this;
            }

            pool_allocator(const pool_allocator& other) noexcept = delete;
            auto operator=(const pool_allocator& other) noexcept -> pool_allocator& = delete;

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type x, size_type y) -> pointer
            {
                if(moved_)
                    return pointer{nullptr};

                if(x_ == 0)
                    x_ = x;

                if(y_ == 0)
                    y_ = y;

                auto ret = static_cast<pointer>(nullptr);

                if(limit_ != 0)
                {
                    while(current_.load() >= limit_)
                        std::this_thread::yield();
                }

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                if(list_.empty())
                    ret = alloc_.allocate(x_, y_);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                ++current_;
                lock_.clear(std::memory_order_release);

                return ret;
            }

            auto allocate_smart(size_type x, size_type y) -> smart_pointer
            {
                auto p = allocate(x, y);
                return smart_pointer{p, [this, p](T*){ this->deallocate(p); }};
            }

            auto deallocate(pointer p, size_type = 0, size_type = 0) noexcept -> void
            {
                if(moved_ || (p == nullptr))
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                --current_;
                lock_.clear(std::memory_order_release);
            }

            auto release() noexcept -> void
            {
                if(moved_) // the allocator becomes invalid once moved from
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, x_, y_);

                x_ = 0;
                y_ = 0;
                current_.store(0);

                lock_.clear(std::memory_order_release);
            }

        private:
            InternalAlloc alloc_;
            std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
            std::forward_list<pointer> list_;
            size_type x_;
            size_type y_;
            size_type limit_;
            std::atomic_size_t current_;
            bool moved_ = false;
    };

    /* 3D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>
    {
        public:
            static constexpr auto mem_layout = InternalAlloc::mem_layout;
            static constexpr auto mem_location = InternalAlloc::mem_location;
            static constexpr auto alloc_needs_pitch = InternalAlloc::alloc_needs_pitch;
            using propagate_on_container_copy_assignment = std::true_type;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;
            using is_always_equal = std::true_type;

            using value_type = T;
            using pointer = typename InternalAlloc::pointer;
            using const_pointer = typename InternalAlloc::const_pointer;
            using size_type = typename InternalAlloc::size_type;
            using difference_type = typename InternalAlloc::difference_type;
            using smart_pointer = typename InternalAlloc::template smart_pointer<std::function<void(T*)>>;

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, mem_layout, InternalAlloc>;
            };

        public:
            pool_allocator() = default;

            pool_allocator(size_type limit)
            : alloc_{}, list_{}, x_{}, y_{}, z_{}, limit_{limit}, current_{0}
            {}

            pool_allocator(pool_allocator&& other) noexcept
            : alloc_{std::move(other.alloc_)}, list_{std::move(other.list_)}
            , x_{other.x_}, y_{other.y_}, z_{other.z_}, limit_{other.limit_}, current_{other.current_.load()}, moved_{other.moved_}
            {
                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;
            }

            auto operator=(pool_allocator&& other) noexcept -> pool_allocator&
            {
                alloc_ = std::move(other.alloc_);
                list_ = std::move(other.list_);
                x_ = other.x_;
                y_ = other.y_;
                z_ = other.z_;
                limit_ = other.limit_;
                current_ = other.current_.load();
                moved_ = other.moved_;
                if(other.lock_.test_and_set())
                    lock_.test_and_set();
                else
                    lock_.clear();

                other.moved_ = true;

                return *this;
            }

            pool_allocator(const pool_allocator& other) noexcept = delete;
            auto operator=(const pool_allocator& other) noexcept -> pool_allocator& = delete;

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type x, size_type y, size_type z) -> pointer
            {
                if(moved_)
                    return pointer{nullptr};

                if(x_ == 0)
                    x_ = x;
                if(y_ == 0)
                    y_ = y;
                if(z_ == 0)
                    z_ = z;

                if(limit_ != 0)
                {
                    while(current_.load() >= limit_)
                        std::this_thread::yield();
                }

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                auto ret = static_cast<pointer>(nullptr);

                if(list_.empty())
                    ret = alloc_.allocate(x_, y_, z_);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                ++current_;
                lock_.clear(std::memory_order_release);

                return ret;
            }

            auto allocate_smart(size_type x, size_type y, size_type z) -> smart_pointer
            {
                auto p = allocate(x, y, z);
                return smart_pointer{p, [this, p](T* ptr){ this->deallocate(p); }};
            }

            auto deallocate(pointer p, size_type = 0, size_type = 0, size_type = 0) noexcept -> void
            {
                if(moved_ || (p == nullptr))
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                --current_;
                lock_.clear(std::memory_order_release);
            }

            auto release() noexcept -> void
            {
                if(moved_) // the allocator becomes invalid once moved from
                    return;

                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, x_, y_, z_);

                x_ = 0;
                y_ = 0;
                z_ = 0;
                current_.store(0);

                lock_.clear(std::memory_order_release);
            }

        private:
            InternalAlloc alloc_;
            std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
            std::forward_list<pointer> list_;
            size_type x_;
            size_type y_;
            size_type z_;
            size_type limit_;
            std::atomic_size_t current_;
            bool moved_ = false;
    };
}


#endif /* DDRF_BITS_POOL_ALLOCATOR_H_ */
