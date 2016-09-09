#ifndef DDRF_CUDA_UTILITY_H_
#define DDRF_CUDA_UTILITY_H_

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/bits/throw_error.h>

namespace ddrf
{
    namespace cuda
    {
        inline auto set_device(int d) -> void
        {
            auto err = cudaSetDevice(d);
            if(err != cudaSuccess)
                detail::throw_error(err);
        }
    }
}



#endif /* DDRF_CUDA_UTILITY_H_ */
