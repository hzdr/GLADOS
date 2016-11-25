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

#ifndef DDRF_CUFFT_EXCEPTION_H_
#define DDRF_CUFFT_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace ddrf
{
    namespace cufft
    {
        /*
         * CUFFT_ALLOC_FAILED
         */
        class bad_alloc : public std::exception
        {
            public:
                bad_alloc() noexcept = default;
                virtual ~bad_alloc() = default;

                auto operator=(const bad_alloc&) noexcept -> bad_alloc& { return *this; }

                virtual auto what() const noexcept -> const char*
                {
                    return "cuFFT failed to allocate GPU or CPU memory";
                }
        };

        /*
         * CUFFT_INVALID_PLAN
         * CUFFT_INVALID_TYPE
         * CUFFT_INVALID_VALUE
         * CUFFT_INVALID_SIZE
         * CUFFT_UNALIGNED_DATA
         * CUFFT_INCOMPLETE_PARAMETER_LIST
         * CUFFT_NO_WORKSPACE
         */
        class invalid_argument : public std::invalid_argument
        {
            public:
                using std::invalid_argument::invalid_argument;
                virtual ~invalid_argument() = default;
        };

        /*
         * CUFFT_INTERNAL_ERROR
         * CUFFT_EXEC_FAILED
         * CUFFT_SETUP_FAILED
         * CUFFT_INVALID_DEVICE
         * CUFFT_PARSE_ERROR
         * CUFFT_NOT_IMPLEMENTED
         * CUFFT_LICENSE_ERROR
         */
        class runtime_error : public std::runtime_error
        {
            public:
                using std::runtime_error::runtime_error;
                virtual ~runtime_error() = default;
        };
    }
}

#endif /* DDRF_CUFFT_EXCEPTION_H_ */
