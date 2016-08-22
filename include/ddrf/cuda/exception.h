#ifndef DDRF_CUDA_EXCEPTION_H_
#define DDRF_CUDA_EXCEPTION_H_

#include <exception>
#include <stdexcept>
#include <string>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ddrf
{
    namespace cuda
    {
        class bad_alloc : public std::exception
        {
            public:
                bad_alloc() noexcept = default;
                virtual ~bad_alloc() = default;

                auto operator=(const bad_alloc& other) noexcept-> bad_alloc& { return *this; }

                virtual auto what() const noexcept -> const char*
                {
                    return cudaGetErrorString(cudaErrorMemoryAllocation);
                }

        };

        class invalid_argument : public std::invalid_argument
        {
            public:
                using std::invalid_argument::invalid_argument;
                virtual ~invalid_argument() = default;
        };

        class runtime_error : public std::runtime_error
        {
            public:
                using std::runtime_error::runtime_error;
                virtual ~runtime_error() = default;
        };
    }
}



#endif /* DDRF_CUDA_EXCEPTION_H_ */
