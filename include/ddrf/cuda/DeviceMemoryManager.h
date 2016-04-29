#ifndef CUDA_DEVICEMEMORYMANAGER_H_
#define CUDA_DEVICEMEMORYMANAGER_H_

#include <climits>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "Memory.h"

namespace ddrf
{
	namespace cuda
	{
		template <class T, class CopyPolicy = sync_copy_policy>
		class DeviceMemoryManager : public CopyPolicy
		{
			public:
				using value_type = T;
				using pointer_type_2D = pitched_device_ptr<T, CopyPolicy, std::false_type>;
				using pointer_type_3D = pitched_device_ptr<T, CopyPolicy, std::true_type>;
				using size_type = std::size_t;

			public:
				DeviceMemoryManager()
				: device_{INT_MIN}
				{
				}

				auto setDevice(int device_id) -> void
				{
					device_ = device_id;
				}

				auto device() -> int
				{
					return device_;
				}

			protected:
				~DeviceMemoryManager() = default;

				auto make_ptr(size_type width, size_type height) -> pointer_type_2D
				{
					return make_device_ptr<value_type, CopyPolicy>(width, height);
				}

				auto make_ptr(size_type width, size_type height, size_type depth) -> pointer_type_3D
				{
					return make_device_ptr<value_type, CopyPolicy>(width, height, depth);
				}

				template <typename Source>
				auto copy(pointer_type_2D& dest, Source& src, size_type width, size_type height) -> void
				{
					CopyPolicy::copy(dest, src, width, height);
				}

				template <typename Source>
				auto copy(pointer_type_3D& dest, Source& src, size_type width, size_type height, size_type depth) -> void
				{
					CopyPolicy::copy(dest, src, width, height, depth);
				}

			private:
				int device_;
		};
	}
}


#endif /* CUDA_DEVICEMEMORYMANAGER_H_ */
