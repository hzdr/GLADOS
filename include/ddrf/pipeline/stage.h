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
 * Date: 11 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_PIPELINE_STAGE_H_
#define GLADOS_PIPELINE_STAGE_H_

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <glados/pipeline/input_side.h>
#include <glados/pipeline/output_side.h>

namespace glados
{
    namespace pipeline
    {
        template <class StageT>
        class stage : public StageT
                    , public input_side<typename StageT::input_type>
                    , public output_side<typename StageT::output_type>
        {
            public:
                using input_type = typename StageT::input_type;
                using output_type = typename StageT::output_type;
                using size_type = std::size_t;

            public:
                template <class... Args>
                stage(Args&&... args)
                : StageT(std::forward<Args>(args)...)
                , input_side<input_type>()
                , output_side<output_type>()
                {}

                template <class... Args>
                stage(size_type input_limit, Args&&... args)
                : StageT(std::forward<Args>(args)...)
                , input_side<input_type>(input_limit)
                , output_side<output_type>()
                {}

                auto run() -> void
                {
                    set_input<input_type>();
                    set_output<output_type>();
                    StageT::run();
                }

            private:
                template <class I>
                auto set_input() const noexcept
                -> typename std::enable_if<std::is_same<void, I>::value && std::is_same<input_type, I>::value, void>::type
                {}

                template <class I>
                auto set_input()
                -> typename std::enable_if<!std::is_same<void, I>::value && std::is_same<input_type, I>::value, void>::type
                {
                    StageT::set_input_function(std::bind(&input_side<input_type>::take, this));
                }

                template <class O>
                auto set_output() const noexcept
                -> typename std::enable_if<std::is_same<void, O>::value && std::is_same<output_type, O>::value, void>::type
                {}

                template <class O>
                auto set_output()
                -> typename std::enable_if<!std::is_same<void, O>::value && std::is_same<output_type, O>::value, void>::type
                {
                    StageT::set_output_function(std::bind(&output_side<output_type>::template output<output_type>, this, std::placeholders::_1));
                }
        };
    }
}

#endif /* GLADOS_PIPELINE_STAGE_H_ */
