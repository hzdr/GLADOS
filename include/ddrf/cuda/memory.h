#ifndef DDRF_CUDA_MEMORY_H_
#define DDRF_CUDA_MEMORY_H_

#include <cstddef>

#ifndef __NVCC__
#include <cuda_runtime.h>
#endif

#include <ddrf/cuda/bits/device_allocator.h>
#include <ddrf/cuda/bits/host_allocator.h>
#include <ddrf/cuda/bits/pitched_ptr.h>
#include <ddrf/cuda/bits/throw_error.h>
#include <ddrf/cuda/bits/unique_ptr.h>

namespace ddrf
{
    namespace cuda
    {
        inline auto get_memory_info(std::size_t& free, std::size_t& total) -> void
        {
            auto err = cudaMemGetInfo(&free, &total);
            if(err != cudaSuccess)
                detail::throw_error(err);
        }
    }
}

#endif /* DDRF_CUDA_MEMORY_H_ */
