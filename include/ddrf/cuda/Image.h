#ifndef CUDA_IMAGE_H_
#define CUDA_IMAGE_H_

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
		template <typename Data, class CopyPolicy = sync_copy_policy>
		class Image
		{
			public:
				using value_type = Data;
				using pointer_type = pitched_device_ptr<Data, CopyPolicy, std::false_type>;
				using size_type = std::size_t;

			public:
				Image()
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
				~Image() = default;

				auto make_ptr(size_type width, size_type height) -> pointer_type
				{
					return make_device_ptr<value_type>(width, height);
				}

				template <typename Source>
				auto copy(pointer_type& dest, Source& src, size_type, size_type) -> void
				{
					dest = src;
				}

			private:
				int device_;
		};
	}
}


#endif /* CUDA_IMAGE_H_ */
