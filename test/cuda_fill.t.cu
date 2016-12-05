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
#include <cstdlib>

#define BOOST_TEST_MODULE CudaFill
#include <boost/test/unit_test.hpp>

#include <glados/cuda/algorithm.h>
#include <glados/cuda/memory.h>
#include <glados/cuda/sync_policy.h>

BOOST_AUTO_TEST_CASE(cuda_fill_sync_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx);
    auto dev = glados::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::sync, dev, dev_val, szx);
    glados::cuda::copy(glados::cuda::sync, host_dest, dev, szx);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_sync_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = glados::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::sync, dev, dev_val, szx, szy);
    glados::cuda::copy(glados::cuda::sync, host_dest, dev, szx, szy);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_sync_3d)
{
    constexpr auto szx = 8;
    constexpr auto szy = 8;
    constexpr auto szz = 8;
    constexpr auto dim = szx * szy * szz;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto dev = glados::cuda::make_unique_device<int>(szx, szy, szz);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::sync, dev, dev_val, szx, szy, szz);
    glados::cuda::copy(glados::cuda::sync, host_dest, dev, szx, szy, szz);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx);
    auto dev = glados::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::async, dev, dev_val, szx);
    cudaDeviceSynchronize();
    glados::cuda::copy(glados::cuda::async, host_dest, dev, szx);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = glados::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::async, dev, dev_val, szx, szy);
    cudaDeviceSynchronize();
    glados::cuda::copy(glados::cuda::async, host_dest, dev, szx, szy);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_3d)
{
    constexpr auto szx = 8;
    constexpr auto szy = 8;
    constexpr auto szz = 8;
    constexpr auto dim = szx * szy * szz;

    auto host_orig = glados::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto host_dest = glados::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto dev = glados::cuda::make_unique_device<int>(szx, szy, szz);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    glados::cuda::fill(glados::cuda::async, dev, dev_val, szx, szy, szz);
    cudaDeviceSynchronize();
    glados::cuda::copy(glados::cuda::async, host_dest, dev, szx, szy, szz);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}
