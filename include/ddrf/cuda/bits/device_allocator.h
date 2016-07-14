#ifndef DDRF_CUDA_BITS_DEVICE_ALLOCATOR_H_
#define DDRF_CUDA_BITS_DEVICE_ALLOCATOR_H_

#include <cstddef>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/bits/location.h>
#include <ddrf/cuda/bits/memory_layout.h>
#include <ddrf/cuda/bits/pitched_ptr.h>

namespace ddrf
{
    namespace cuda
    {
        /*
         * FIXME: C++14 support
         *  Once C++14 is supported by CUDA, add the following type:
         *      using propagate_on_container_move_assignment = std::true_type
         *
         * FIXME: C++17 support
         *  Once C++17 is supported by CUDA, add the following type:
         *      using is_always_equal = std::true_type
         */
        template <class T, memory_layout ml>
        class device_allocator {};

        template <class T>
        class device_allocator<T, memory_layout::pointer_1D>
        {
            private:
                static constexpr auto ml = memory_layout::pointer_1D;

            public:
                static constexpr auto memory_location = location::device;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, ml>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
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
                    auto err = cudaMemset(reinterpret_cast<void*>(p), value, n);
                    if(err != cudaSuccess)
                        throw invalid_argument{cudaGetErrorString(err)};
                }
        };

        template <class T>
        class device_allocator<T, memory_layout::pointer_2D>
        {
            private:
                static constexpr auto ml = memory_layout::pointer_2D;

            public:
                static constexpr auto memory_location = location::device;
                static constexpr auto alloc_needs_pitch = true;

                using value_type = T;
                using pointer = pitched_ptr<value_type>;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, ml>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
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
                    auto err = cudaFree(reinterpret_cast<void*>(p.ptr));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y) -> void
                {
                    auto err = cudaMemset2D(reinterpret_cast<void*>(p.ptr), p.pitch, value, x * sizeof(value_type), y);
                    if(err != cudaSuccess)
                        throw invalid_argument{cudaGetErrorString(err)};
                }
        };

        template <class T>
        class device_allocator<T, memory_layout::pointer_3D>
        {
            private:
                static constexpr auto ml = memory_layout::pointer_3D;

            public:
                static constexpr auto memory_location = location::device;
                static constexpr auto alloc_needs_pitch = true;

                using value_type = T;
                using pointer = pitched_ptr<value_type>;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;

                template <class U>
                struct rebind
                {
                    using other = device_allocator<U, ml>;
                };

                device_allocator() noexcept = default;
                device_allocator(const device_allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                device_allocator(const device_allocator<U, uml>& other) noexcept
                {
                    static_assert(std::is_same<T, U>::value && ml == uml, "Attempting to copy incompatible device allocator");
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
                    auto err = cudaFree(reinterpret_cast<void*>(p.ptr));
                    if(err != cudaSuccess)
                        std::exit(err);
                }

                auto fill(pointer p, int value, size_type x, size_type y, size_type z) -> void
                {
                    auto extent = make_cudaExtent(x * sizeof(value_type), y, z);
                    auto pitched_ptr = make_cudaPitchedPtr(reinterpret_cast<void*>(p.ptr), p.pitch, x, y);
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
