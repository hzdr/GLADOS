#ifndef DDRF_CUDA_BITS_HOST_ALLOCATOR_H_
#define DDRF_CUDA_BITS_HOST_ALLOCATOR_H_

#include <algorithm>
#include <cstddef>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/bits/location.h>
#include <ddrf/cuda/bits/memory_layout.h>

namespace ddrf
{
    namespace cuda
    {
        template <class T, memory_layout ml = memory_layout::pointer_1D>
        class host_allocator {};

        template <class T>
        class host_allocator<T, memory_layout::pointer_1D>
        {
            public:
                static constexpr auto memory_location = location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = host_allocator<U, ml>;
                };

                host_allocator() noexcept = default;
                host_allocator(const host_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                host_allocator(const host_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
                }

                ~host_allocator() = default;

                auto allocate(size_type n) -> pointer
                {
                    auto p = static_cast<pointer>(nullptr);
                    if(cudaMallocHost(reinterpret_cast<void**>(&p), n * sizeof(value_type)) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{p};
                }

                auto deallocate(pointer p, size_type = 0) noexcept -> void
                {
                    auto err = cudaFreeHost(reinterpret_cast<void*>(p));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type n) -> void
                {
                    std::fill(p, p + n, value);
                }
        };

        template <class T>
        class host_allocator<T, memory_layout::pointer_2D>
        {
            public:
                static constexpr auto memory_location = location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = host_allocator<U, ml>;
                };

                host_allocator() noexcept = default;
                host_allocator(const host_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                host_allocator(const host_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
                }

                ~host_allocator() = default;

                auto allocate(size_type x, size_type y) -> pointer
                {
                    auto p = static_cast<pointer>(nullptr);
                    if(cudaMallocHost(reinterpret_cast<void**>(&p), x * y * sizeof(value_type)) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{p};
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0) noexcept -> void
                {
                    auto err = cudaFreeHost(reinterpret_cast<void*>(p));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y) -> void
                {
                    std::fill(p, p + (x * y), value);
                }
        };

        template <class T>
        class host_allocator<T, memory_layout::pointer_3D>
        {
            public:
                static constexpr auto memory_location = location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = host_allocator<U, ml>;
                };

                host_allocator() noexcept = default;
                host_allocator(const host_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                host_allocator(const host_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
                }

                ~host_allocator() = default;

                auto allocate(size_type x, size_type y, size_type z) -> pointer
                {
                    auto p = static_cast<pointer>(nullptr);
                    if(cudaMallocHost(reinterpret_cast<void**>(&p), x * y * z * sizeof(value_type)) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{p};
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0, size_type = 0) noexcept -> void
                {
                    auto err = cudaFreeHost(reinterpret_cast<void*>(p));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y, size_type z) -> void
                {
                    std::fill(p, p + (x * y * z), value);
                }
        };

        template <class T1, memory_layout ml1, class T2, memory_layout ml2>
        auto operator==(const host_allocator<T1, ml1>& lhs, const host_allocator<T2, ml2>& rhs) noexcept -> bool
        {
            return true;
        }

        template <class T1, memory_layout ml1, class T2, memory_layout ml2>
        auto operator!=(const host_allocator<T1, ml1>& lhs, const host_allocator<T2, ml2>& rhs) noexcept -> bool
        {
            return false;
        }
    }
}



#endif /* DDRF_CUDA_BITS_HOST_ALLOCATOR_H_ */
