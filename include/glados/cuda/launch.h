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
            constexpr auto dim_x = 16u;
            constexpr auto dim_y = 16u;
            auto block_size = dim3{dim_x, dim_y}; // for whatever reason we can't make this constexpr

            auto blocks_x = detail::round_up(input_width, dim_x) / dim_x;
            auto blocks_y = detail::round_up(input_height, dim_y) / dim_y;
            
            auto grid_size = dim3{blocks_x, blocks_y};
            kernel<<<grid_size, block_size, 0, stream>>>(args...);
            auto err = cudaPeekAtLastError();
            if(err != cudaSuccess)
                detail::throw_error(err);
        }

        template <typename... Args>
        auto launch_async(cudaStream_t stream, std::size_t input_width, std::size_t input_height, std::size_t input_depth, void(*kernel)(Args...), Args... args) -> void
        {
            constexpr auto dim_x = 16u;
            constexpr auto dim_y = 16u;
            constexpr auto dim_z = 2u;
            auto block_size = dim3{dim_x, dim_y, dim_z};

            auto blocks_x = detail::round_up(input_width, dim_x) / dim_x;
            auto blocks_y = detail::round_up(input_height, dim_y) / dim_y;
            auto blocks_z = detail::round_up(input_depth, dim_z) / dim_z;
            
            auto grid_size = dim3{blocks_x, blocks_y, blocks_z};
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
