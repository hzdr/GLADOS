#ifndef DDRF_CUDA_MEMORY_POOL_H_
#define DDRF_CUDA_MEMORY_POOL_H_

#include <atomic>
#include <cstddef>
#include <forward_list>
#include <iterator>
#include <map>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddrf/cuda/memory.h>

namespace ddrf
{
    namespace cuda
    {
        template <class T, memory_layout ml, class Alloc = device_allocator<T, ml>>
        class memory_pool {};

        template <class T, class Alloc>
        class memory_pool<T, memory_layout::pointer_1D, Alloc>
        {
            public:
                using allocator_type = typename std::allocator_traits<Alloc>::allocator_type;
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using difference_type = typename std::allocator_traits<Alloc>::difference_type;
                using size_type = typename std::allocator_traits<Alloc>::size_type;

                class pool_deleter
                {
                    public:
                        pool_deleter(memory_pool* pool) : pool_{pool} {}
                        auto operator()(pointer ptr) noexcept -> void { pool_->release(ptr); }

                    private:
                        memory_pool* pool_;
                };

                using pooled_object = unique_ptr<T, pool_deleter, Alloc::alloc_needs_pitch, Alloc::memory_location>;

                memory_pool(size_type limit, size_type w)
                : alloc_{}, locked_{false}, list_{limit, nullptr}, ptr_width_{w}
                {}

                memory_pool(const memory_pool& other)
                : alloc_{other.alloc_}, list_{other.list_}, ptr_width_{other.ptr_width_}
                {
                    locked_.store(other.locked_.load());
                }

                ~memory_pool()
                {
                    for(auto&& p : list_)
                        alloc_.deallocate(p);
                }

                auto acquire() -> pooled_object
                {
                    while(list_.empty() || locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    auto ptr = std::move(list_.front());
                    list_.pop_front();
                    locked_ = false;

                    if(ptr == nullptr)
                    {
                        // the correct overload is chosen at compile time with enable_if
                        ptr = alloc_.allocate(ptr_width_);
                        alloc_.fill(ptr, 0, ptr_width_);
                    }

                    return pooled_object{ptr, pool_deleter{this}};
                }

                auto release(pointer ptr) -> void
                {
                    if(ptr == nullptr)
                        return;

                    alloc_.fill(ptr, 0, ptr_width_);

                    while(locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    list_.push_front(ptr);
                    locked_ = false;
                }

                auto release(pooled_object obj) -> void
                {
                    release(obj.release());
                }

            private:
                allocator_type alloc_;

                std::atomic_bool locked_;
                std::forward_list<pointer> list_;
                size_type ptr_width_;
        };

        template <class T, class Alloc>
        class memory_pool<T, memory_layout::pointer_2D, Alloc>
        {
            public:
                using allocator_type = typename std::allocator_traits<Alloc>::allocator_type;
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using difference_type = typename std::allocator_traits<Alloc>::difference_type;
                using size_type = typename std::allocator_traits<Alloc>::size_type;

                class pool_deleter
                {
                    public:
                        pool_deleter(memory_pool* pool) : pool_{pool} {}
                        auto operator()(pointer ptr) noexcept -> void { pool_->release(ptr); }

                    private:
                        memory_pool* pool_;
                };

                using pooled_object = unique_ptr<T, pool_deleter, Alloc::alloc_needs_pitch, Alloc::memory_location>;

                memory_pool(size_type limit, size_type w, size_type h)
                : alloc_{}, locked_{false}, list_{limit, nullptr}, ptr_width_{w}, ptr_height_{h}
                {}

                memory_pool(const memory_pool& other)
                : alloc_{other.alloc_}, list_{other.list_}, ptr_width_{other.ptr_width_}
                , ptr_height_{other.ptr_height_}
                {
                    locked_.store(other.locked_.load());
                }

                ~memory_pool()
                {
                    for(auto&& p : list_)
                        alloc_.deallocate(p);
                }

                auto acquire() -> pooled_object
                {
                    while(list_.empty() || locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    auto ptr = std::move(list_.front());
                    list_.pop_front();
                    locked_ = false;

                    if(ptr == nullptr)
                    {
                        // the correct overload is chosen at compile time with enable_if
                        ptr = alloc_.allocate(ptr_width_, ptr_height_);
                        alloc_.fill(ptr, 0, ptr_width_, ptr_height_);
                    }

                    return pooled_object{ptr, pool_deleter{this}};
                }

                auto release(pointer ptr) -> void
                {
                    if(ptr == nullptr)
                        return;

                    alloc_.fill(ptr, 0, ptr_width_, ptr_height_);

                    while(locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    list_.push_front(ptr);
                    locked_ = false;
                }

                auto release(pooled_object obj) -> void
                {
                    release(pointer{obj.release(), obj.pitch()});
                }

            private:
                allocator_type alloc_;

                std::atomic_bool locked_;
                std::forward_list<pointer> list_;
                size_type ptr_width_;
                size_type ptr_height_;
        };

        template <class T, class Alloc>
        class memory_pool<T, memory_layout::pointer_3D, Alloc>
        {
            public:
                using allocator_type = typename std::allocator_traits<Alloc>::allocator_type;
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using difference_type = typename std::allocator_traits<Alloc>::difference_type;
                using size_type = typename std::allocator_traits<Alloc>::size_type;

                class pool_deleter
                {
                    public:
                        pool_deleter(memory_pool* pool) : pool_{pool} {}
                        auto operator()(pointer ptr) noexcept -> void { pool_->release(ptr); }

                    private:
                        memory_pool* pool_;
                };

                using pooled_object = unique_ptr<T, pool_deleter, Alloc::alloc_needs_pitch, Alloc::memory_location>;

                memory_pool(size_type limit, size_type w, size_type h, size_type d)
                : alloc_{}, locked_{false}, list_{limit, nullptr}, ptr_width_{w}, ptr_height_{h}, ptr_depth_{d}
                {}

                memory_pool(const memory_pool& other)
                : alloc_{other.alloc_}, list_{other.list_}, ptr_width_{other.ptr_width_}
                , ptr_height_{other.ptr_height_}, ptr_depth_{other.ptr_depth_}
                {
                    locked_.store(other.locked_.load());
                }

                ~memory_pool()
                {
                    for(auto&& p : list_)
                        alloc_.deallocate(p);
                }

                auto acquire() -> pooled_object
                {
                    while(list_.empty() || locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    auto ptr = std::move(list_.front());
                    list_.pop_front();
                    locked_ = false;

                    if(ptr == nullptr)
                    {
                        // the correct overload is chosen at compile time with enable_if
                        ptr = alloc_.allocate(ptr_width_, ptr_height_, ptr_depth_);
                        alloc_.fill(ptr, 0, ptr_width_, ptr_height_, ptr_depth_);
                    }

                    return pooled_object{ptr, pool_deleter{this}};
                }

                auto release(pointer ptr) -> void
                {
                    if(ptr == nullptr)
                        return;

                    alloc_.fill(ptr, 0, ptr_width_, ptr_height_, ptr_depth_);

                    while(locked_)
                        std::this_thread::yield();

                    locked_ = true;
                    list_.push_front(ptr);
                    locked_ = false;
                }

                auto release(pooled_object obj) -> void
                {
                    release(pointer{obj.release(), obj.pitch()});
                }

            private:
                allocator_type alloc_;

                std::atomic_bool locked_;
                std::forward_list<pointer> list_;
                size_type ptr_width_;
                size_type ptr_height_;
                size_type ptr_depth_;
        };
    }
}

#endif /* DDRF_CUDA_MEMORY_POOL_H_ */
