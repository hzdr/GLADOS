#ifndef DDRF_CUDA_BITS_MEMCPY_DIRECTION_H_
#define DDRF_CUDA_BITS_MEMCPY_DIRECTION_H_

#include <ddrf/bits/memory_location.h>

namespace ddrf
{
    namespace cuda
    {
        namespace detail
        {
            template <memory_location d, memory_location s>
            struct memcpy_direction {};

            template <>
            struct memcpy_direction<memory_location::device, memory_location::host>
            {
                static constexpr auto value = cudaMemcpyHostToDevice;
            };

            template <>
            struct memcpy_direction<memory_location::host, memory_location::device>
            {
                static constexpr auto value = cudaMemcpyDeviceToHost;
            };

            template <>
            struct memcpy_direction<memory_location::host, memory_location::host>
            {
                static constexpr auto value = cudaMemcpyHostToHost;
            };

            template <>
            struct memcpy_direction<memory_location::device, memory_location::device>
            {
                static constexpr auto value = cudaMemcpyDeviceToDevice;
            };
        }
    }
}



#endif /* DDRF_CUDA_BITS_MEMCPY_DIRECTION_H_ */
