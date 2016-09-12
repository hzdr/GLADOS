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

        inline auto get_device() -> int
        {
            auto d = int{};
            auto err = cudaGetDevice(&d);
            if(err != cudaSuccess)
                detail::throw_error(err);

            return d;
        }

        inline auto get_device_count() -> int
        {
            auto d = int{};
            auto err = cudaGetDeviceCount(&d);
            if(err != cudaSuccess)
                detail::throw_error(err);

            return d;
        }

        inline auto create_stream() -> cudaStream_t
        {
            auto s = cudaStream_t{};
            auto err = cudaStreamCreate(&s);
            if(err != cudaSuccess)
                detail::throw_error(err);

            return s;
        }

        inline auto synchronize_stream(cudaStream_t stream = 0) -> void
        {
            auto err = cudaStreamSynchronize(stream);
            if(err != cudaSuccess)
                detail::throw_error(err);
        }
    }
}



#endif /* DDRF_CUDA_UTILITY_H_ */
