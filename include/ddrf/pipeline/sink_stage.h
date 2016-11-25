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
 * Date: 11 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef DDRF_PIPELINE_SINK_STAGE_H_
#define DDRF_PIPELINE_SINK_STAGE_H_

#include <functional>
#include <utility>

#include <ddrf/pipeline/input_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class SinkT>
        class sink_stage : public SinkT
                         , public input_side<typename SinkT::input_type>
        {
            public:
                using input_type = typename SinkT::input_type;
                using size_type = typename input_side<input_type>::size_type;

            public:
                template <class... Args>
                sink_stage(Args&&... args)
                : SinkT(std::forward<Args>(args)...), input_side<input_type>()
                {}

                template <class... Args>
                sink_stage(size_type input_limit, Args&&... args)
                : SinkT(std::forward<Args>(args)...), input_side<input_type>(input_limit)
                {}

                auto run() -> void
                {
                    SinkT::set_input_function(std::bind(&input_side<input_type>::take, this));
                    SinkT::run();
                }
        };
    }
}



#endif /* DDRF_PIPELINE_SINK_STAGE_H_ */
