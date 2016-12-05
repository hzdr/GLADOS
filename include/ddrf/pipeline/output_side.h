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
 * Date: 14 July 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_PIPELINE_OUTPUT_SIDE_H_
#define GLADOS_PIPELINE_OUTPUT_SIDE_H_

#include <type_traits>
#include <utility>

#include <glados/pipeline/input_side.h>

namespace glados
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

#endif /* GLADOS_PIPELINE_OUTPUT_SIDE_H_ */
