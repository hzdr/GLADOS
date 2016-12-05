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
 * Date: 12 July 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_CUDA_MEMORY_H_
#define GLADOS_CUDA_MEMORY_H_

#include <cstddef>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

#include <glados/cuda/bits/device_allocator.h>
#include <glados/cuda/bits/host_allocator.h>
#include <glados/cuda/bits/pitched_ptr.h>
#include <glados/cuda/bits/throw_error.h>
#include <glados/cuda/bits/unique_ptr.h>

namespace glados
{
    namespace cuda
    {
        inline auto get_memory_info(std::size_t& free, std::size_t& total) -> void
        {
            auto err = cudaMemGetInfo(&free, &total);
            if(err != cudaSuccess)
                detail::throw_error(err);
        }
    }
}

#endif /* GLADOS_CUDA_MEMORY_H_ */
