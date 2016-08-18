#ifndef DDRF_CUDA_BITS_DEVICE_ALLOCATOR_H_
#define DDRF_CUDA_BITS_DEVICE_ALLOCATOR_H_

#include <cstddef>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/bits/memory_layout.h>
#include <ddrf/bits/memory_location.h>
#include <ddrf/cuda/bits/pitched_ptr.h>
#include <ddrf/cuda/bits/unique_ptr.h>
#include <ddrf/cuda/exception.h>


namespace ddrf
{
    namespace cuda
    {
        template <class T, memory_layout ml>
        class device_allocator {};

        template <class T>
        class device_allocator<T, memory_layout::pointer_1D>
        {
            public:
                static constexpr auto memory_layout = memory_layout::pointer_1D;
                static constexpr auto memory_location = memory_location::device;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = unique_ptr<T, Deleter, alloc_needs_pitch, memory_location, false>;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, memory_layout>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, ddrf::memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && memory_layout == uml, "Attempting to copy incompatible device allocator");
                }

                ~device_allocator() = default;

                auto allocate(size_type n) -> pointer
                {
                    auto ptr = static_cast<pointer>(nullptr);
                    if(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(value_type)) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{ptr};
                }

                auto deallocate(pointer p, size_type = 0) noexcept -> void
                {
                    auto err = cudaFree(reinterpret_cast<void*>(p));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type n) -> void
                {
                    constexpr auto size = sizeof(T);
                    auto err = cudaMemset(p, value, n * size);
                    if(err != cudaSuccess)
                        throw invalid_argument{cudaGetErrorString(err)};
                }
        };

        template <class T>
        class device_allocator<T, memory_layout::pointer_2D>
        {
            public:
                static constexpr auto memory_layout = memory_layout::pointer_2D;
                static constexpr auto memory_location = memory_location::device;
                static constexpr auto alloc_needs_pitch = true;

                using value_type = T;
                using pointer = pitched_ptr<value_type>;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = unique_ptr<T, Deleter, alloc_needs_pitch, memory_location, false>;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, memory_layout>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, ddrf::memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && memory_layout == uml, "Attempting to copy incompatible device allocator");
                }

                ~device_allocator() = default;

                auto allocate(size_type x, size_type y) -> pointer
                {
                    auto ptr = static_cast<pointer>(nullptr);
                    auto pitch = size_type{};
                    if(cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, x * sizeof(value_type), y) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{ptr, pitch};
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0) noexcept -> void
                {
                    auto err = cudaFree(reinterpret_cast<void*>(p.ptr()));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y) -> void
                {
                    constexpr auto size = sizeof(T);
                    auto err = cudaMemset2D(p.ptr(), p.pitch(), value, x * size, y);
                    if(err != cudaSuccess)
                        throw invalid_argument{cudaGetErrorString(err)};
                }
        };

        template <class T>
        class device_allocator<T, memory_layout::pointer_3D>
        {
            public:
                static constexpr auto memory_layout = memory_layout::pointer_3D;
                static constexpr auto memory_location = memory_location::device;
                static constexpr auto alloc_needs_pitch = true;

                using value_type = T;
                using pointer = pitched_ptr<value_type>;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = unique_ptr<T, Deleter, alloc_needs_pitch, memory_location, false>;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, memory_layout>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, ddrf::memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && memory_layout == uml, "Attempting to copy incompatible device allocator");
                }

                ~device_allocator() = default;

                auto allocate(size_type x, size_type y, size_type z) -> pointer
                {
                    auto extent = make_cudaExtent(x * sizeof(value_type), y, z);
                    auto pitched_ptr = cudaPitchedPtr{};
                    if(cudaMalloc3D(&pitched_ptr, extent) == cudaErrorMemoryAllocation)
                        throw bad_alloc{};
                    return pointer{pitched_ptr.ptr, pitched_ptr.pitch};
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0, size_type = 0) noexcept -> void
                {
                    auto err = cudaFree(reinterpret_cast<void*>(p.ptr()));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y, size_type z) -> void
                {
                    constexpr auto size = sizeof(T);
                    auto extent = make_cudaExtent(x * size, y, z);
                    auto pitched_ptr = make_cudaPitchedPtr(p.ptr(), p.pitch(), x * size, y);

                    auto err = cudaMemset3D(pitched_ptr, value, extent);
                    if(err != cudaSuccess)
                        throw invalid_argument{cudaGetErrorString(err)};
                }
        };

        template <class T1, memory_layout ml1, class T2, memory_layout ml2>
        auto operator==(const device_allocator<T1, ml1>& lhs, const device_allocator<T2, ml2>& rhs) noexcept -> bool
        {
            return ml1 == ml2;
        }

        template <class T1, memory_layout ml1, class T2, memory_layout ml2>
        auto operator!=(const device_allocator<T1, ml1>& lhs, const device_allocator<T2, ml2>& rhs) noexcept -> bool
        {
            return ml1 != ml2;
        }
    }
}

#endif /* DDRF_CUDA_BITS_DEVICE_ALLOCATOR_H_ */
