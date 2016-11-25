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

#ifndef DDRF_CUSPARSE_EXCEPTION_H_
#define DDRF_CUSPARSE_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace ddrf
{
    namespace cusparse
    {
        /*
         * CUSPARSE_STATUS_ALLOC_FAILED
         */
        class bad_alloc : public std::exception
        {
            bad_alloc() noexcept = default;
            virtual ~bad_alloc() = default;

            auto operator=(const bad_alloc&) noexcept -> bad_alloc& { return *this; }
            auto what() const noexcept -> const char*
            {
                return "Resource allocation failed inside the cuSPARSE library";
            }
        }

        /*
         * CUSPARSE_STATUS_INVALID_VALUE
         * CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
         * CUSPARSE_STATUS_ZERO_PIVOT
         */
        class invalid_argument : public std::invalid_argument
        {
            public:
                using std::invalid_argument::invalid_argument;
                virtual ~invalid_argument() = default;
        };

        /*
         * CUSPARSE_STATUS_NOT_INITIALIZED
         * CUSPARSE_STATUS_ARCH_MISMATCH
         * CUSPARSE_STATUS_MAPPING_ERROR
         * CUSPARSE_STATUS_EXECUTION_FAILED
         * CUSPARSE_STATUS_INTERNAL_ERROR
         */
        class runtime_error : public std::runtime_error
        {
            public:
                using std::runtime_error::runtime_error;
                virtual ~runtime_error() = default;
        };
    }
}



#endif /* DDRF_CUSPARSE_EXCEPTION_H_ */
