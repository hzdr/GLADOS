#ifndef DEF_MEMORYMANAGER_H_
#define DEF_MEMORYMANAGER_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "Memory.h"

namespace ddrf
{
	namespace def
	{
		template <class T>
		class MemoryManager
		{
			public:
				using value_type = T;
				using size_type = std::size_t;
				using pointer_type_2D = pitched_ptr<T, std::false_type>;
				using pointer_type_3D = pitched_ptr<T, std::true_type>;

			public:
				inline auto make_ptr(size_type width, size_type height) -> pointer_type_2D
				{
					auto ptr = std::unique_ptr<T[]>(new value_type[width * height]);
					return pitched_ptr<T, std::false_type>(std::move(ptr), width * sizeof(T), width, height);
				}

				inline auto make_ptr(size_type width, size_type height, size_type depth) -> pointer_type_3D
				{
					auto ptr = std::unique_ptr<T>(new value_type[width * height * depth]);
					return pitched_ptr<T, std::true_type>(std::move(ptr), width * sizeof(T), width, height, depth);
				}

				inline auto copy(pointer_type_2D& dest, const pointer_type_2D& src, size_type width, size_type height) -> void
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}

				inline auto copy(pointer_type_3D& dest, const pointer_type_3D& src, size_type width, size_type height, size_type depth) -> void
				{
					std::copy(src.get(), src.get() + width * height * depth, dest.get());
				}
		};
	}
}

#endif /* DEF_MEMORYMANAGER_H_ */
