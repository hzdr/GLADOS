#include <algorithm>
#include <cstdlib>

#define BOOST_TEST_MODULE CudaFill
#include <boost/test/unit_test.hpp>

#include <ddrf/cuda/algorithm.h>
#include <ddrf/cuda/memory.h>
#include <ddrf/cuda/sync_policy.h>

BOOST_AUTO_TEST_CASE(cuda_fill_sync_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto dev = ddrf::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::sync, dev, dev_val, szx);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_sync_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::sync, dev, dev_val, szx, szy);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx, szy);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_sync_3d)
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

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::sync, dev, dev_val, szx, szy, szz);
    ddrf::cuda::copy(ddrf::cuda::sync, host_dest, dev, szx, szy, szz);

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_1d)
{
    constexpr auto szx = 4096;
    constexpr auto dim = szx;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx);
    auto dev = ddrf::cuda::make_unique_device<int>(szx);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::async, dev, dev_val, szx);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_2d)
{
    constexpr auto szx = 64;
    constexpr auto szy = 64;
    constexpr auto dim = szx * szy;

    auto host_orig = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto host_dest = ddrf::cuda::make_unique_pinned_host<int>(szx, szy);
    auto dev = ddrf::cuda::make_unique_device<int>(szx, szy);

    auto ho = host_orig.get();
    auto hd = host_dest.get();

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::async, dev, dev_val, szx, szy);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx, szy);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}

BOOST_AUTO_TEST_CASE(cuda_fill_async_3d)
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

    constexpr auto host_val = 0x01010101;
    constexpr auto dev_val = 0x01;

    std::fill(ho, ho + dim, host_val);
    std::generate(hd, hd + dim, std::rand);

    ddrf::cuda::fill(ddrf::cuda::async, dev, dev_val, szx, szy, szz);
    cudaDeviceSynchronize();
    ddrf::cuda::copy(ddrf::cuda::async, host_dest, dev, szx, szy, szz);
    cudaDeviceSynchronize();

    BOOST_CHECK(std::equal(ho, ho + dim, hd));
}
