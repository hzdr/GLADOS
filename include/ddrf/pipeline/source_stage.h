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

#ifndef DDRF_PIPELINE_SOURCE_STAGE_H_
#define DDRF_PIPELINE_SOURCE_STAGE_H_

#include <functional>
#include <utility>

#include <ddrf/pipeline/output_side.h>

namespace ddrf
{
    namespace pipeline
    {
        template <class SourceT>
        class source_stage : public SourceT
                           , public output_side<typename SourceT::output_type>
        {
            public:
                using output_type = typename SourceT::output_type;

            public:
                template <class... Args>
                source_stage(Args&&... args)
                : SourceT(std::forward<Args>(args)...), output_side<output_type>()
                {}

                auto run() -> void
                {
                    SourceT::set_output_function(std::bind(&output_side<output_type>::template output<output_type>, this, std::placeholders::_1));
                    SourceT::run();
                }
        };
    }
}

#endif /* DDRF_PIPELINE_SOURCE_STAGE_H_ */
