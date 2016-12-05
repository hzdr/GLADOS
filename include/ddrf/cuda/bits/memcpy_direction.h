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

#ifndef GLADOS_CUDA_BITS_MEMCPY_DIRECTION_H_
#define GLADOS_CUDA_BITS_MEMCPY_DIRECTION_H_

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <glados/bits/memory_location.h>

namespace glados
{
    namespace cuda
    {
        namespace detail
        {
            template <memory_location d, memory_location s>
            struct memcpy_direction {};

            template <>
            struct memcpy_direction<memory_location::device, memory_location::host>
            {
                static constexpr auto value = cudaMemcpyHostToDevice;
            };

            template <>
            struct memcpy_direction<memory_location::host, memory_location::device>
            {
                static constexpr auto value = cudaMemcpyDeviceToHost;
            };

            template <>
            struct memcpy_direction<memory_location::host, memory_location::host>
            {
                static constexpr auto value = cudaMemcpyHostToHost;
            };

            template <>
            struct memcpy_direction<memory_location::device, memory_location::device>
            {
                static constexpr auto value = cudaMemcpyDeviceToDevice;
            };
        }
    }
}

#endif /* GLADOS_CUDA_BITS_MEMCPY_DIRECTION_H_ */
