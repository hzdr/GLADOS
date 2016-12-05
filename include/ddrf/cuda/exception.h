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

#ifndef GLADOS_CUDA_EXCEPTION_H_
#define GLADOS_CUDA_EXCEPTION_H_

#include <exception>
#include <stdexcept>
#include <string>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace glados
{
    namespace cuda
    {
        class bad_alloc : public std::exception
        {
            public:
                bad_alloc() noexcept = default;
                virtual ~bad_alloc() = default;

                auto operator=(const bad_alloc&) noexcept-> bad_alloc& { return *this; }

                virtual auto what() const noexcept -> const char*
                {
                    return cudaGetErrorString(cudaErrorMemoryAllocation);
                }

        };

        class invalid_argument : public std::invalid_argument
        {
            public:
                using std::invalid_argument::invalid_argument;
                virtual ~invalid_argument() = default;
        };

        class runtime_error : public std::runtime_error
        {
            public:
                using std::runtime_error::runtime_error;
                virtual ~runtime_error() = default;
        };
    }
}

#endif /* GLADOS_CUDA_EXCEPTION_H_ */
