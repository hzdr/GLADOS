/*
 * This file is part of the GLADOS library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * GLADOS is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GLADOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with GLADOS. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Date: 06 September 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <thread>

#define BOOST_TEST_MODULE CudaUninitialized
#include <boost/test/unit_test.hpp>

#include <glados/cuda/algorithm.h>
#include <glados/cuda/coordinates.h>
#include <glados/cuda/launch.h>
#include <glados/cuda/memory.h>
#include <glados/cuda/sync_policy.h>
#include <glados/memory.h>

__global__ void kernel(float *dst, std::size_t d_p, const float* src, std::size_t s_p, std::size_t w, std::size_t h)
{
    auto x = glados::cuda::coord_x();
    auto y = glados::cuda::coord_y();

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

    auto src = glados::cuda::make_unique_device<float>(w, h);
    auto dst = glados::cuda::make_unique_device<float>(w, h);

    glados::cuda::fill(glados::cuda::sync, src, 0x01, w, h);
    glados::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
    cudaDeviceSynchronize();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_pool)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    using device_allocator = glados::cuda::device_allocator<float, glados::memory_layout::pointer_2D>;
    using pool_allocator = glados::pool_allocator<float, glados::memory_layout::pointer_2D, device_allocator>;
    auto alloc = pool_allocator{};

    auto src = alloc.allocate_smart(w, h);
    auto dst = alloc.allocate_smart(w, h);

    glados::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
    cudaDeviceSynchronize();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_make_unique_stream)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    auto t = std::thread{[w, h]() {
            auto src = glados::cuda::make_unique_device<float>(w, h);
            auto dst = glados::cuda::make_unique_device<float>(w, h);

            glados::cuda::fill(glados::cuda::sync, src, 0x01, w, h);
            glados::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
            cudaDeviceSynchronize();
        }
    };
    t.join();
}

BOOST_AUTO_TEST_CASE(cuda_uninitialized_pool_stream)
{
    constexpr auto w = std::size_t{1024};
    constexpr auto h = std::size_t{768};

    using device_allocator = glados::cuda::device_allocator<float, glados::memory_layout::pointer_2D>;
    using pool_allocator = glados::pool_allocator<float, glados::memory_layout::pointer_2D, device_allocator>;
    auto t = std::thread{[w, h]() {
            auto alloc = pool_allocator{};

            auto src = alloc.allocate_smart(w, h);
            auto dst = alloc.allocate_smart(w, h);

            glados::cuda::launch(w, h, kernel, dst.get(), dst.pitch(), static_cast<const float*>(src.get()), src.pitch(), w, h);
            cudaDeviceSynchronize();
        }
    };
    t.join();
}
