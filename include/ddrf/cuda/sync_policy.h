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
 * Date: 15 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_CUDA_SYNC_POLICY_H_
#define GLADOS_CUDA_SYNC_POLICY_H_

#include <algorithm>
#include <cstddef>
#include <exception>
#include <future>
#include <thread>
#include <utility>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <glados/bits/memory_location.h>
#include <glados/cuda/bits/memcpy_direction.h>
#include <glados/cuda/bits/throw_error.h>
#include <glados/cuda/exception.h>

namespace glados
{
    namespace cuda
    {
        namespace detail
        {
            template <class D, class S>
            auto create_3D_parms(D& d, const S& s, std::size_t x, std::size_t y, std::size_t z,
                    std::size_t d_off_x, std::size_t d_off_y, std::size_t d_off_z,
                    std::size_t s_off_x, std::size_t s_off_y, std::size_t s_off_z) -> cudaMemcpy3DParms
                    {
                        constexpr auto uchar_size = sizeof(unsigned char);
                        constexpr auto d_elem_size = sizeof(typename D::element_type) / uchar_size;
                        constexpr auto s_elem_size = sizeof(typename S::element_type) / uchar_size;

                        auto to_uchar_pos = [](std::size_t v, std::size_t size) { return v * size; };
                        auto x_ext = to_uchar_pos(x, d_elem_size);

                        auto extent = make_cudaExtent(x_ext, y, z);

                        constexpr auto d_size = sizeof(typename D::element_type);
                        constexpr auto s_size = sizeof(typename S::element_type);

                        auto d_pitch = d.pitch();
                        if(D::mem_location == memory_location::host)
                            d_pitch = x * d_size;

                        auto s_pitch = s.pitch();
                        if(S::mem_location == memory_location::host)
                            s_pitch = x * s_size;

                        auto d_pitched = make_cudaPitchedPtr(d.get(), d_pitch, x, y);
                        auto s_pitched = make_cudaPitchedPtr(s.get(), s_pitch, x, y);

                        auto d_pos_x = to_uchar_pos(d_off_x, d_elem_size);
                        auto d_pos_y = to_uchar_pos(d_off_y, d_elem_size);
                        auto d_pos_z = to_uchar_pos(d_off_z, d_elem_size);
                        auto s_pos_x = to_uchar_pos(s_off_x, s_elem_size);
                        auto s_pos_y = to_uchar_pos(s_off_y, s_elem_size);
                        auto s_pos_z = to_uchar_pos(s_off_z, s_elem_size);

                        auto d_pos = make_cudaPos(d_pos_x, d_pos_y, d_pos_z);
                        auto s_pos = make_cudaPos(s_pos_x, s_pos_y, s_pos_z);

                        auto parms = cudaMemcpy3DParms{};
                        parms.srcPos = s_pos;
                        parms.srcPtr = s_pitched;
                        parms.dstPos = d_pos;
                        parms.dstPtr = d_pitched;
                        parms.extent = extent;
                        parms.kind = detail::memcpy_direction<D::mem_location, S::mem_location>::value;

                        return parms;
            }
        }

        class sync_policy
        {
            public:
                template <class D, class S>
                auto copy(D& d, const S& s, std::size_t x) const -> void
                {
                    static_assert(!D::pitched_memory, "Destination memory must not be pitched for a 1D copy.");
                    static_assert(!S::pitched_memory, "Source memory must not be pitched for a 1D copy.");

                    constexpr auto size = sizeof(typename D::element_type);

                    auto err = cudaMemcpy(d.get(), s.get(), x * size, detail::memcpy_direction<D::mem_location, S::mem_location>::value);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class D, class S>
                auto copy(D& d, const S& s, std::size_t x, std::size_t y) const -> void
                {
                    static_assert((D::mem_location == memory_location::host) || D::pitched_memory, "Destination memory on the device must be pitched for a 2D copy.");
                    static_assert((S::mem_location == memory_location::host) || S::pitched_memory, "Source memory on the device must be pitched for a 2D copy.");

                    constexpr auto size = sizeof(typename D::element_type);

                    auto d_pitch = d.pitch();
                    if(D::mem_location == memory_location::host)
                        d_pitch = x * size;

                    auto s_pitch = s.pitch();
                    if(S::mem_location == memory_location::host)
                        s_pitch = x * size;

                    auto err = cudaMemcpy2D(d.get(), d_pitch, s.get(), s_pitch, x * size, y, detail::memcpy_direction<D::mem_location, S::mem_location>::value);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class D, class S>
                auto copy(D& d, const S& s, std::size_t x, std::size_t y, std::size_t z,
                            std::size_t d_off_x = 0, std::size_t d_off_y = 0, std::size_t d_off_z = 0,
                            std::size_t s_off_x = 0, std::size_t s_off_y = 0, std::size_t s_off_z = 0) const
                -> void
                {
                    static_assert((D::mem_location == memory_location::host) || D::pitched_memory, "Destination memory on the device must be pitched for a 3D copy.");
                    static_assert((S::mem_location == memory_location::host) || S::pitched_memory, "Source memory on the device must be pitched for a 3D copy.");

                    auto parms = detail::create_3D_parms(d, s, x, y, z, d_off_x, d_off_y, d_off_z, s_off_x, s_off_y, s_off_z);
                    auto err = cudaMemcpy3D(&parms);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(!P::pitched_memory, "The memory on the device must not be pitched for a 1D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);
                    auto err = cudaMemset(p.get(), value, x * size);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    std::fill_n(p.get(), x, value);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(P::pitched_memory, "The memory on the device must be pitched for a 2D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);
                    auto err = cudaMemset2D(p.get(), p.pitch(), value, x * size, y);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    std::fill_n(p.get(), x * y, value);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y, std::size_t z) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(P::pitched_memory, "The memory on the device must be pitched for a 3D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);
                    auto extent = make_cudaExtent(x * size, y, z);
                    auto pitched_ptr = make_cudaPitchedPtr(p.get(), p.pitch(), x * size, y);

                    auto err = cudaMemset3D(pitched_ptr, value, extent);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y, std::size_t z) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    std::fill_n(p.get(), x * y * z, value);
                }
        };

        class async_policy
        {
            public:
                template <class D, class S>
                auto copy(D& d, const S& s, cudaStream_t stream, std::size_t x) const -> void
                {
                    static_assert(!D::pitched_memory, "Destination memory must not be pitched for a 1D copy");
                    static_assert(!S::pitched_memory, "Source memory must not be pitched for a 1D copy");
                    static_assert((D::mem_location == memory_location::device) || D::pinned_memory, "Destination on host memory must be pinned for asynchronous copies.");
                    static_assert((S::mem_location == memory_location::device) || S::pinned_memory, "Source on host memory must be pinned for asynchronous copies.");

                    constexpr auto size = sizeof(typename D::element_type);

                    auto err = cudaMemcpyAsync(d.get(), s.get(), x * size, detail::memcpy_direction<D::mem_location, S::mem_location>::value, stream);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class D, class S>
                auto copy(D& d, const S& s, cudaStream_t stream, std::size_t x, std::size_t y) const -> void
                {
                    static_assert((D::mem_location == memory_location::host) || D::pitched_memory, "Destination memory on the device must be pitched for a 2D copy.");
                    static_assert((S::mem_location == memory_location::host) || S::pitched_memory, "Source memory on the device must be pitched for a 2D copy.");
                    static_assert((D::mem_location == memory_location::device) || D::pinned_memory, "Destination memory on the host must be pinned for asynchronous copies.");
                    static_assert((S::mem_location == memory_location::device) || S::pinned_memory, "Source memory on the host must be pinned for asynchronous copies.");

                    constexpr auto size = sizeof(typename D::element_type);

                    auto d_pitch = d.pitch();
                    if(D::mem_location == memory_location::host)
                        d_pitch = x * size;

                    auto s_pitch = s.pitch();
                    if(S::mem_location == memory_location::host)
                        s_pitch = x * size;

                    auto err = cudaMemcpy2DAsync(d.get(), d_pitch, s.get(), s_pitch, x * size, y, detail::memcpy_direction<D::mem_location, S::mem_location>::value, stream);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class D, class S>
                auto copy(D& d, const S& s, cudaStream_t stream, std::size_t x, std::size_t y, std::size_t z,
                        std::size_t d_off_x = 0, std::size_t d_off_y = 0, std::size_t d_off_z = 0,
                        std::size_t s_off_x = 0, std::size_t s_off_y = 0, std::size_t s_off_z = 0) const -> void
                {
                    static_assert((D::mem_location == memory_location::host) || D::pitched_memory, "Destination memory on the device must be pitched for a 3D copy.");
                    static_assert((S::mem_location == memory_location::host) || S::pitched_memory, "Source memory on the host must be pitched for a 3D copy.");
                    static_assert((D::mem_location == memory_location::device) || D::pinned_memory, "Destination memory on the host must be pinned for asynchronous copies.");
                    static_assert((S::mem_location == memory_location::device) || S::pinned_memory, "Source memory on the host must be pinned for asynchronous copies.");

                    auto parms = detail::create_3D_parms(d, s, x, y, z, d_off_x, d_off_y, d_off_z, s_off_x, s_off_y, s_off_z);

                    auto err = cudaMemcpy3DAsync(&parms, stream);
                    if(err != cudaSuccess)

                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, cudaStream_t stream, std::size_t x) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(!P::pitched_memory, "The memory on the device must not be pitched for a 1D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);

                    auto err = cudaMemsetAsync(p.get(), value, x * size, stream);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    // in an ideal world we wouldn't spawn threads manually but use std::async instead.
                    // However, the world isn't ideal and std::async is completely useless when you want to launch "fire and forget" tasks.
                    auto f = [&]() { std::fill_n(p.get(), x, value); };
                    auto&& t = std::thread{f};
                    t.detach();
                }

                template <class P>
                auto fill(P& p, int value, cudaStream_t stream, std::size_t x, std::size_t y) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(P::pitched_memory, "The memory on the device must be pitched for a 2D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);

                    auto err = cudaMemset2DAsync(p.get(), p.pitch(), value, x * size, y, stream);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    auto f = [&]() { std::fill_n(p.get(), x * y, value); };
                    auto&& t = std::thread{f};
                    t.detach();
                }

                template <class P>
                auto fill(P& p, int value, cudaStream_t stream, std::size_t x, std::size_t y, std::size_t z) const
                -> typename std::enable_if<P::mem_location == memory_location::device, void>::type
                {
                    static_assert(P::pitched_memory, "The memory on the device must be pitched for a 3D fill operation.");

                    constexpr auto size = sizeof(typename P::element_type);
                    auto extent = make_cudaExtent(x * size, y, z);
                    auto pitched_ptr = make_cudaPitchedPtr(p.get(), p.pitch(), x * size, y);

                    auto err = cudaMemset3DAsync(pitched_ptr, value, extent, stream);
                    if(err != cudaSuccess)
                        detail::throw_error(err);
                }

                template <class P>
                auto fill(P& p, int value, std::size_t x, std::size_t y, std::size_t z) const
                -> typename std::enable_if<P::mem_location == memory_location::host, void>::type
                {
                    auto f = [&](){ std::fill_n(p.get(), x * y * z, value); };
                    auto&& t = std::thread{f};
                    t.detach();
                }
        };

        constexpr auto sync = sync_policy{};
        constexpr auto async = async_policy{};
    }
}

#endif /* GLADOS_CUDA_SYNC_POLICY_H_ */
