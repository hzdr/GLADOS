#ifndef DEF_MEMORYMANAGER_H_
#define DEF_MEMORYMANAGER_H_

#include <algorithm>
#include <cstddef>
#include <memory>

#include "Memory.h"

namespace ddrf
{
	namespace def
	{
		template <class T, class CopyPolicy = copy_policy>
		class MemoryManager
		{
			public:
				using value_type = T;
				using size_type = std::size_t;
				using pointer_type = pitched_ptr<T>;

			public:
				inline auto make_ptr(size_type width, size_type height) -> pointer_type
				{
					return std::unique_ptr<T>(new value_type[width * height]);
				}

				inline auto copy(pointer_type& dest, const pointer_type& src, size_type width, size_type height) -> void
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}
		};
	}
}

#endif /* DEF_MEMORYMANAGER_H_ */
