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

                auto operator=(const bad_alloc& other) noexcept-> bad_alloc&;
                virtual auto what() const noexcept -> const char*
                {
                    return cudaGetErrorString(cudaErrorMemoryAllocation);
                }

        };

        class invalid_argument : public std::invalid_argument
        {
            public:
                explicit invalid_argument(const std::string& what_arg)
                : std::invalid_argument{what_arg}
                {}

                explicit invalid_argument(const char* what_arg)
                : std::invalid_argument{what_arg}
                {}

                virtual ~invalid_argument() = default;
        };
    }
}



#endif /* DDRF_CUDA_EXCEPTION_H_ */
