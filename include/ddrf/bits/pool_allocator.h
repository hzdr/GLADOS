#ifndef DDRF_BITS_POOL_ALLOCATOR_H_
#define DDRF_BITS_POOL_ALLOCATOR_H_

#include <atomic>
#include <forward_list>
#include <thread>
#include <type_traits>
#include <utility>

#include <ddrf/bits/memory_layout.h>

namespace ddrf
{
    template <class T, memory_layout ml, class InternalAlloc, class = typename std::enable_if<(ml == InternalAlloc::layout)>::type>
    class pool_allocator {};

    /* 1D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>
    {
        public:
            static constexpr auto memory_layout = InternalAlloc::memory_layout;
            static constexpr auto memory_location = InternalAlloc::memory_location;
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

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, memory_layout, InternalAlloc>;
            };

        public:
            pool_allocator() noexcept = default;
            pool_allocator(const pool_allocator& other) noexcept = default;

            template <class U, ddrf::memory_layout uml>
            pool_allocator(const pool_allocator& other) noexcept
            {
                static_assert(std::is_same<T, U>::value && (memory_layout == uml), "Attempting to copy incompatible pool allocator");
            }

            auto operator=(const pool_allocator& other) noexcept -> pool_allocator&
            {
                return *this;
            }

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type n) -> pointer
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                if(n_ == 0)
                    n_ = n;

                auto ret = static_cast<pointer>(nullptr);

                if(list_.empty())
                    ret = alloc_.allocate(n_);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                lock_.clear(std::memory_order_release);

                fill(ret, 0);
                return ret;
            }

            auto deallocate(pointer p, size_type = 0) noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                lock_.clear(std::memory_order_release);
            }

            auto fill(pointer p, int value, size_type = 0) -> void
            {
                alloc_.fill(p, value, n_);
            }

            auto release() noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, n_);

                n_ = 0;

                lock_.clear(std::memory_order_release);
            }

        private:
            static InternalAlloc alloc_;
            static std::atomic_flag lock_;
            static std::forward_list<pointer> list_;
            static size_type n_;
    };

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>::alloc_ = InternalAlloc();

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>::lock_ = ATOMIC_FLAG_INIT;

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>::list_ = std::forward_list<pointer>{100u, nullptr};

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_1D, InternalAlloc>::n_ = size_type{0};

    /* 2D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>
    {
        public:
            static constexpr auto layout = memory_layout::pointer_2D;
            static constexpr auto location = InternalAlloc::location;
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

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, layout, InternalAlloc>;
            };

        public:
            pool_allocator() noexcept = default;
            pool_allocator(const pool_allocator& other) noexcept = default;

            template <class U, memory_layout uml>
            pool_allocator(const pool_allocator& other) noexcept
            {
                static_assert(std::is_same<T, U>::value && (layout == uml), "Attempting to copy incompatible pool allocator");
            }

            auto operator=(const pool_allocator& other) noexcept -> pool_allocator&
            {
                return *this;
            }

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type x, size_type y) -> pointer
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                auto ret = static_cast<pointer>(nullptr);

                if(list_.empty())
                    ret = alloc_.allocate(x, y);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                lock_.clear(std::memory_order_release);
                return ret;
            }

            auto deallocate(pointer p, size_type = 0, size_type = 0) noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                lock_.clear(std::memory_order_release);
            }

            auto fill(pointer p, int value, size_type x, size_type y) -> void
            {
                alloc_.fill(p, value, x, y);
            }

            auto release() noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, x_, y_);

                x_ = 0;
                y_ = 0;

                lock_.clear(std::memory_order_release);
            }

        private:
            static InternalAlloc alloc_;
            static std::atomic_flag lock_;
            static std::forward_list<pointer> list_;
            static size_type x_;
            static size_type y_;
    };

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>::alloc_ = InternalAlloc();

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>::lock_ = ATOMIC_FLAG_INIT;

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>::list_ = std::forward_list<pointer>{100u, nullptr};

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>::x_ = size_type{0};

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_2D, InternalAlloc>::y_ = size_type{0};

    /* 3D specialization */
    template <class T, class InternalAlloc>
    class pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>
    {
        public:
            static constexpr auto layout = memory_layout::pointer_3D;
            static constexpr auto location = InternalAlloc::location;
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

            template <class U>
            struct rebind
            {
                using other = pool_allocator<U, layout, InternalAlloc>;
            };

        public:
            pool_allocator() noexcept = default;
            pool_allocator(const pool_allocator& other) noexcept
            {
            }

            template <class U, memory_layout uml>
            pool_allocator(const pool_allocator& other) noexcept
            {
                static_assert(std::is_same<T, U>::value && (layout == uml), "Attempting to copy incompatible pool allocator");
            }

            auto operator=(const pool_allocator& other) noexcept -> pool_allocator&
            {
                return *this;
            }

            ~pool_allocator()
            {
                // pool_allocator's contents have to be released manually
            }

            auto allocate(size_type x, size_type y, size_type z) -> pointer
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                if(x_ == 0)
                    x_ = x;
                if(y_ == 0)
                    y_ = y;
                if(z_ == 0)
                    z_ = z;

                auto ret = static_cast<pointer>(nullptr);

                if(list_.empty())
                    ret = alloc_.allocate(x_, y_, z_);
                else
                {
                    ret = std::move(list_.front());
                    list_.pop_front();
                }

                lock_.clear(std::memory_order_release);

                fill(ret, 0);
                return ret;
            }

            auto deallocate(pointer p, size_type = 0, size_type = 0, size_type = 0) noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                list_.push_front(p);
                lock_.clear(std::memory_order_release);
            }

            auto fill(pointer p, int value, size_type = 0, size_type = 0, size_type = 0) -> void
            {
                alloc_.fill(p, value, x_, y_, z_);
            }

            auto release() noexcept -> void
            {
                while(lock_.test_and_set(std::memory_order_acquire))
                    std::this_thread::yield();

                for(auto&& p : list_)
                    alloc_.deallocate(p, x_, y_, z_);

                x_ = 0;
                y_ = 0;
                z_ = 0;

                lock_.clear(std::memory_order_release);
            }

        private:
            static InternalAlloc alloc_;
            static std::atomic_flag lock_;
            static std::forward_list<pointer> list_;
            static size_type x_;
            static size_type y_;
            static size_type z_;

    };

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::alloc_ = InternalAlloc();

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::lock_ = ATOMIC_FLAG_INIT;

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::list_ = std::forward_list<typename pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::pointer>{100u, nullptr};

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::x_ = 0;

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::y_ = 0;

    template <class T, class InternalAlloc>
    auto pool_allocator<T, memory_layout::pointer_3D, InternalAlloc>::z_ = 0;
}


#endif /* DDRF_BITS_POOL_ALLOCATOR_H_ */
