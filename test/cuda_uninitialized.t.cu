#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <thread>

#define BOOST_TEST_MODULE CudaUninitialized
#include <boost/test/unit_test.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/coordinates.h>
#include <ddrf/cuda/launch.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>
#include <ddrf/memory.h>

__global__ void kernel(float *dst, std::size_t d_p, const float* src, std::size_t s_p, std::size_t w, std::size_t h)
{
    auto x = ddrf::cuda::coord_x();
    auto y = ddrf::cuda::coord_y();

    if((x < w) && (y < h))
    {
        auto src_row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src) + s_p * y);
        auto dst_row = reinterpret_cast<float*>(reinterpret_cast<char*>(dst) + d_p * y);

        dst_row[x] = src_row[x] + 42;
    }
}


BOOST_AUTO_TEST_CASE(cuda_uninitialized_make_unique)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    auto src = ddrf::cuda::make_unique_device<float>(w, h);
    auto dst = ddrf::cuda::make_unique_device<float>(w, h);

    ddrf::cuda::fill(ddrf::cuda::sync, src, 0x01, w, h);
    ddrf::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
    cudaDeviceSynchronize();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_pool)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
    using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
    auto alloc = pool_allocator{};

    auto src = alloc.allocate_smart(w, h);
    auto dst = alloc.allocate_smart(w, h);

    ddrf::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
    cudaDeviceSynchronize();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_make_unique_stream)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    auto t = std::thread{[w, h]() {
            auto src = ddrf::cuda::make_unique_device<float>(w, h);
            auto dst = ddrf::cuda::make_unique_device<float>(w, h);

            ddrf::cuda::fill(ddrf::cuda::sync, src, 0x01, w, h);
            ddrf::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
            cudaDeviceSynchronize();
        }
    };
    t.join();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_pool_stream)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    using device_allocator = ddrf::cuda::device_allocator<float, ddrf::memory_layout::pointer_2D>;
    using pool_allocator = ddrf::pool_allocator<float, ddrf::memory_layout::pointer_2D, device_allocator>;
    auto t = std::thread{[w, h]() {
            auto alloc = pool_allocator{};

            auto src = alloc.allocate_smart(w, h);
            auto dst = alloc.allocate_smart(w, h);

            ddrf::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
            cudaDeviceSynchronize();
        }
    };
    t.join();
}
