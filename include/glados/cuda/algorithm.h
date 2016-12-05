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
 * Date: 15 August 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_CUDA_ALGORITHM_H_
#define GLADOS_CUDA_ALGORITHM_H_

#include <cstddef>
#include <utility>
#include <type_traits>

namespace glados
{
    namespace cuda
    {
        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& policy, D& dst, const S& src, Args&&... args) -> void
        {
            policy.copy(dst, src, std::forward<Args>(args)...);
        }

        /**
         * Note that fill() will apply value to the individual bytes of the data, not the elements
         */
        template <class SyncPolicy, class P, class... Args>
        auto fill(SyncPolicy&& policy, P& p, int value, Args&&... args) -> void
        {
            policy.fill(p, value, std::forward<Args>(args)...);
        }
    }
}

#endif /* GLADOS_CUDA_ALGORITHM_H_ */
