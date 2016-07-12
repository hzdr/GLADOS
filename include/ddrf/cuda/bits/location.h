#ifndef DDRF_CUDA_BITS_LOCATION_H_
#define DDRF_CUDA_BITS_LOCATION_H_

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ddrf
{
    namespace cuda
    {
        enum class location { device, host };

        template<location src, location dst> struct direction {};
        template<> struct direction<location::host, location::device> { static constexpr auto value = cudaMemcpyHostToDevice; };
        template<> struct direction<location::host, location::host> { static constexpr auto value = cudaMemcpyHostToHost; };
        template<> struct direction<location::device, location::host> { static constexpr auto value = cudaMemcpyDeviceToHost; };
        template<> struct direction<location::device, location::device> { static constexpr auto value = cudaMemcpyDeviceToDevice; };
    }
}



#endif /* LOCATION_H_ */
