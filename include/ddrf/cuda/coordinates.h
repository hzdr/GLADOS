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
 * Date: 15 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDRF_CUDA_COORDINATES_H_
#define DDRF_CUDA_COORDINATES_H_

namespace ddrf
{
    namespace cuda
    {
        inline __device__ auto coord_x() -> unsigned int
        {
            return blockIdx.x * blockDim.x + threadIdx.x;
        }

        inline __device__ auto coord_y() -> unsigned int
        {
            return blockIdx.y * blockDim.y + threadIdx.y;
        }

        inline __device__ auto coord_z() -> unsigned int
        {
            return blockIdx.z * blockDim.z + threadIdx.z;
        }
    }
}

#endif /* DDRF_CUDA_COORDINATES_H_ */
