#ifndef MEMORY_H_
#define MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <memory>

#include "../Memory.h"

namespace ddrf
{
	namespace def
	{
		class copy_policy
		{
			protected:
				~copy_policy() = default;

				/* 1D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t size) -> void
				{
					std::copy(src.get(), src.get() + size, dest.get());
				}

				/* 2D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height)
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}

				/* 3D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height)
				{
					std::copy(src.get(), src.get() + width * height, dest.get());
				}
		};

		template <class T> using ptr = ddrf::ptr<T, copy_policy, std::unique_ptr<T[]>>;
		template <class T, class is3D> using pitched_ptr = ddrf::pitched_ptr<T, copy_policy, is3D, std::unique_ptr<T[]>>;
	}
}



#endif /* MEMORY_H_ */
