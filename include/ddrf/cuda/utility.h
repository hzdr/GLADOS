#ifndef DDRF_CUDA_UTILITY_H_
#define DDRF_CUDA_UTILITY_H_

#include <cstddef>
#include <vector>

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

        inline auto get_device_properties(int device) -> cudaDeviceProp
        {
            auto prop = cudaDeviceProp{};
            auto err = cudaGetDeviceProperties(&prop, device);
            if(err != cudaSuccess)
                detail::throw_error(err);

            return prop;
        }

        inline auto set_valid_devices(int* devices, std::size_t len) -> void
        {
            auto len_i = static_cast<int>(len);
            auto err = cudaSetValidDevices(devices, len_i);
            if(err != cudaSuccess)
                detail::throw_error(err);
        }

        inline auto set_valid_devices(std::vector<int>& devices) -> void
        {
            set_valid_devices(devices.data(), devices.size());
        }

        inline auto create_stream() -> cudaStream_t
        {
            auto s = cudaStream_t{};
            auto err = cudaStreamCreate(&s);
            if(err != cudaSuccess)
                detail::throw_error(err);

            return s;
        }

        inline auto create_concurrent_stream() -> cudaStream_t
        {
            auto s = cudaStream_t{};
            auto err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
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
