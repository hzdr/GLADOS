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

#ifndef GLADOS_CUDA_LAUNCH_H_
#define GLADOS_CUDA_LAUNCH_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include <glados/cuda/bits/throw_error.h>
#include <glados/cuda/exception.h>

namespace glados
{
    namespace cuda
    {
        namespace detail
        {
            inline auto round_up(std::uint32_t num, std::uint32_t multiple) -> std::uint32_t
            {
                if(multiple == 0)
                    return num;

                if(num == 0)
                    return multiple;

                auto remainder = num % multiple;
                if(remainder == 0)
                    return num;

                return num + multiple - remainder;
            }
        }

        template <typename... Args>
        auto launch_async(cudaStream_t stream, std::size_t input_size, void(*kernel)(Args...), Args... args) -> void
        {
            // calculate max potential blocks
            auto block_size = int{};
            auto min_grid_size = int{};
            auto err = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);
            if(err != cudaSuccess)
                detail::throw_error(err);

            // calculate de facto occupation based on input size
            auto block_size_u = static_cast<unsigned int>(block_size);
            auto grid_size = (static_cast<unsigned int>(input_size) + block_size_u - 1u) / block_size_u;

            kernel<<<grid_size, block_size_u, 0u, stream>>>(std::forward<Args>(args)...);
            err = cudaPeekAtLastError();
            if(err != cudaSuccess)
                detail::throw_error(err);
        }

        template <typename... Args>
        auto launch_async(cudaStream_t stream, std::size_t input_width, std::size_t input_height, void(*kernel)(Args...), Args... args) -> void
        {
            auto threads = detail::round_up(static_cast<unsigned int>(input_width * input_height), 1024u);
            auto blocks = threads / 1024u;

            auto iwb = static_cast<unsigned int>(input_width) / blocks;
            auto dim_x = ((iwb < 32u) && (iwb != 0u)) ? iwb : detail::round_up(iwb, 32u);
            auto ihb = static_cast<unsigned int>(input_height) / blocks;
            auto dim_y = ((ihb < 32u) && (ihb != 0u)) ? ihb : detail::round_up(ihb, 32u);
            auto block_size = dim3{dim_x, dim_y};
            auto grid_size = dim3{static_cast<unsigned int>((input_width + block_size.x - 1u)/block_size.x),
                                    static_cast<unsigned int>((input_height + block_size.y - 1u)/block_size.y)};

            kernel<<<grid_size, block_size, 0, stream>>>(args...);
            auto err = cudaPeekAtLastError();
            if(err != cudaSuccess)
                detail::throw_error(err);
        }

        template <typename... Args>
        auto launch_async(cudaStream_t stream, std::size_t input_width, std::size_t input_height, std::size_t input_depth, void(*kernel)(Args...), Args... args) -> void
        {
            auto threads = detail::round_up(static_cast<unsigned int>(input_width * input_height * input_depth), 1024u);
            auto blocks = threads / 1024u;

            auto iwb = static_cast<unsigned int>(input_width) / blocks;
            auto dim_x = ((iwb < 16u) && (iwb != 0u)) ? iwb : detail::round_up(iwb, 16u);
            auto ihb = static_cast<unsigned int>(input_height) / blocks;
            auto dim_y = ((ihb < 16u) && (ihb != 0u)) ? ihb : detail::round_up(ihb, 16u);
            auto idb = static_cast<unsigned int>(input_depth) / blocks;
            auto dim_z = ((idb < 4u) && (idb != 0u)) ? idb : detail::round_up(idb, 4u);
            auto block_size = dim3{dim_x, dim_y, dim_z};
            auto grid_size = dim3{static_cast<unsigned int>((input_width + block_size.x - 1) / block_size.x),
                                    static_cast<unsigned int>((input_height + block_size.y - 1) / block_size.y),
                                    static_cast<unsigned int>((input_depth + block_size.z - 1) / block_size.z)};

            kernel<<<grid_size, block_size, 0, stream>>>(args...);
            auto err = cudaPeekAtLastError();
            if(err != cudaSuccess)
                detail::throw_error(err);
        }

        template <typename... Args>
        auto launch(std::size_t input_width, void (*kernel)(Args...), Args... args) -> void
        {
            launch_async(0, input_width, kernel, std::forward<Args>(args)...);
        }

        template <typename... Args>
        auto launch(std::size_t input_width, std::size_t input_height, void (*kernel)(Args...), Args... args) -> void
        {
            launch_async(0, input_width, input_height, kernel, std::forward<Args>(args)...);
        }

        template <typename... Args>
        auto launch(std::size_t input_width, std::size_t input_height, std::size_t input_depth, void (*kernel)(Args...), Args... args) -> void
        {
            launch_async(0, input_width, input_height, input_depth, kernel, std::forward<Args>(args)...);
        }
    }
}

#endif /* GLADOS_CUDA_LAUNCH_H_ */