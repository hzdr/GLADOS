#ifndef CUDA_HOSTMEMORYMANAGER_H_
#define CUDA_HOSTMEMORYMANAGER_H_

#include <cstddef>
#include <type_traits>

#include "Memory.h"

namespace ddrf
{
	namespace cuda
	{
		template <class T, class CopyPolicy = sync_copy_policy>
		class HostMemoryManager : public CopyPolicy
		{
			public:
				using value_type = T;
				using pointer_type_2D = pitched_host_ptr<T, CopyPolicy, std::false_type>;
				using pointer_type_3D = pitched_host_ptr<T, CopyPolicy, std::true_type>;
				using size_type = std::size_t;

			protected:
				inline auto make_ptr(size_type width, size_type height) -> pointer_type_2D
				{
					return make_host_ptr<value_type>(width, height);
				}

				inline auto make_ptr(size_type width, size_type height, size_type depth) -> pointer_type_3D
				{
					return make_host_ptr<value_type>(width, height, depth);
				}

				template <typename Source>
				inline auto copy(pointer_type_2D& dest, Source& src, size_type width, size_type height) -> void
				{
					CopyPolicy::copy(dest, src, width, height);
				}

				template <typename Source>
				inline auto copy(pointer_type_3D& dest, Source& src, size_type width, size_type height, size_type depth) -> void
				{
					CopyPolicy::copy(dest, src, width, height, depth);
				}

				~HostMemoryManager() = default;
		};
	}
}




#endif /* CUDA_HOSTMEMORYMANAGER_H_ */
