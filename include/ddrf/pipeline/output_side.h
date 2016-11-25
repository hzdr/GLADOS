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
 * Date: 14 July 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDRF_PIPELINE_OUTPUT_SIDE_H_
#define DDRF_PIPELINE_OUTPUT_SIDE_H_

#include <type_traits>
#include <utility>

#include <ddrf/pipeline/input_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class OutputT>
        class output_side
        {
            public:
                template <class T>
                auto output(T&& t)
                -> typename std::enable_if<std::is_same<T, OutputT>::value, void>::type
                {
                    if(next_ == nullptr)
                        return;

                    next_->input(std::forward<T>(t));
                }

                auto attach(input_side<OutputT>* next) noexcept
                -> void
                {
                    next_ = next;
                }

            private:
                input_side<OutputT>* next_;
        };

        template <>
        class output_side<void>
        {
        };
    }
}



#endif /* DDRF_PIPELINE_OUTPUT_SIDE_H_ */
