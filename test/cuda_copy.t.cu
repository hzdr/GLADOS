/*
 * This file is part of the ddrf library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * ddrf is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ddrf is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ddrf. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Date: 06 September 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#include <algorithm>
#include <cstdlib>

#define BOOST_TEST_MODULE CudaCopy
#include <boost/test/unit_test.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>

BOOST_AUTO_TEST_CASE(cuda_copy_sync_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto dev = ddrf::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::sync, dev, host_orig, szx);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_copy_sync_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::sync, dev, host_orig, szx, szy);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx, szy);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_copy_sync_3d)
{
    constexpr auto szx = 8;
    constexpr auto szy = 8;
    constexpr auto szz = 8;
    constexpr auto dim = szx * szy * szz;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy, szz);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::sync, dev, host_orig, szx, szy, szz);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx, szy, szz);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_copy_async_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto dev = ddrf::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::async, dev, host_orig, szx);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_copy_async_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::async, dev, host_orig, szx, szy);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx, szy);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_copy_async_3d)
{
    constexpr auto szx = 8;
    constexpr auto szy = 8;
    constexpr auto szz = 8;
    constexpr auto dim = szx * szy * szz;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy, szz);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy, szz);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    std::generate(ho, ho + dim, std::rand);
    std::fill(hd, hd + dim, 0);

    ddrf::cuda::copy(ddrf::cuda::async, dev, host_orig, szx, szy, szz);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx, szy, szz);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}
