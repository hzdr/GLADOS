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
 * Date: 05 December 2016
 * Authors: Jan Stephan <j.stephan@hzdr.de>
 */

#ifndef GLADOS_GENERIC_ALLOCATOR_H_
#define GLADOS_GENERIC_ALLOCATOR_H_

#include <algorithm>
#include <memory>
#include <cstddef>

#include <glados/bits/memory_layout.h>
#include <glados/bits/memory_location.h>

namespace glados
{
    namespace generic
    {
        template <class T, memory_layout ml>
        class allocator {};

        template <class T>
        class allocator<T, memory_layout::pointer_1D>
        {
            public:
                static constexpr auto mem_layout = memory_layout::pointer_1D;
                static constexpr auto mem_location = memory_location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = std::unique_ptr<T[], Deleter>;

                template <class U>
                struct rebind
                {
                    using other = allocator<U, mem_layout>;
                };

                allocator() noexcept = default;
                allocator(const allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                allocator(const allocator<U, uml>&) noexcept
                {
                    static_assert(std::is_same<T, U>::value && mem_layout == uml, "Attempting to copy incompatible allocator");
                }

                ~allocator() = default;

                auto allocate(size_type n) -> pointer
                {
                    return new T[n];
                }

                auto deallocate(pointer p, size_type = 0) noexcept -> void
                {
                    delete[] p;
                }
        };

        template <class T>
        class allocator<T, memory_layout::pointer_2D>
        {
            public:
                static constexpr auto mem_layout = memory_layout::pointer_2D;
                static constexpr auto mem_location = memory_location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = std::unique_ptr<T[], Deleter>;

                template <class U>
                struct rebind
                {
                    using other = allocator<U, mem_layout>;
                };

                allocator() noexcept = default;
                allocator(const allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                allocator(const allocator<U, uml>&) noexcept
                {
                    static_assert(std::is_same<T, U>::value && mem_layout == uml, "Attempting to copy incompatible allocator");
                }

                ~allocator() = default;

                auto allocate(size_type x, size_type y) -> pointer
                {
                    return new T[x * y];
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0) noexcept -> void
                {
                    delete[] p;
                }
        };

        template <class T>
        class allocator<T, memory_layout::pointer_3D>
        {
            public:
                static constexpr auto mem_layout = memory_layout::pointer_3D;
                static constexpr auto mem_location = memory_location::host;
                static constexpr auto alloc_needs_pitch = false;

                using value_type = T;
                using pointer = value_type*;
                using const_pointer = const pointer;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap = std::true_type;
                using is_always_equal = std::true_type;

                template <class Deleter>
                using smart_pointer = std::unique_ptr<T[], Deleter>;

                template <class U>
                struct rebind
                {
                    using other = allocator<U, mem_layout>;
                };

                allocator() noexcept = default;
                allocator(const allocator& other) noexcept = default;

                template <class U, memory_layout uml>
                allocator(const allocator<U, uml>&) noexcept
                {
                    static_assert(std::is_same<T, U>::value && mem_layout == uml, "Attempting to copy incompatible allocator");
                }

                ~allocator() = default;

                auto allocate(size_type x, size_type y, size_type z) -> pointer
                {
                    return new T[x * y * z];
                }

                auto deallocate(pointer p, size_type = 0, size_type = 0, size_type = 0) noexcept -> void
                {
                    delete[] p;
                }
        };
    }
}
    
#endif /* GLADOS_GENERIC_ALLOCATOR_H_ */
