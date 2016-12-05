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

#ifndef GLADOS_CUDA_BITS_PITCHED_PTR_H_
#define GLADOS_CUDA_BITS_PITCHED_PTR_H_

#include <cstddef>

namespace glados
{
    namespace cuda
    {
        template <typename T>
        class pitched_ptr
        {
            public:
                explicit pitched_ptr(T* p, std::size_t ptr_pitch) noexcept : ptr_{p}, pitch_{ptr_pitch} {}
                explicit pitched_ptr(std::nullptr_t) noexcept : ptr_{nullptr}, pitch_{0} {}

                auto ptr() const noexcept -> T* { return ptr_; }
                auto pitch() const noexcept -> std::size_t { return pitch_; }

            private:
                T* ptr_;
                std::size_t pitch_;
        };

        template <class T>
        auto operator==(const pitched_ptr<T>& x, std::nullptr_t) -> bool
        {
            return x.ptr() == nullptr;
        }

        template <class T>
        auto operator==(std::nullptr_t, const pitched_ptr<T>& y) -> bool
        {
            return nullptr == y.ptr();
        }

        template <class T>
        auto operator!=(const pitched_ptr<T>&x, std::nullptr_t) -> bool
        {
            return !(x == nullptr);
        }

        template <class T>
        auto operator!=(std::nullptr_t, const pitched_ptr<T>& y) -> bool
        {
            return !(nullptr == y);
        }
    }
}

#endif /* GLADOS_CUDA_BITS_PITCHED_PTR_H_ */
