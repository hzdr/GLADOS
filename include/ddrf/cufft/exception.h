#ifndef DDRF_CUFFT_EXCEPTION_H_
#define DDRF_CUFFT_EXCEPTION_H_

#include <exception>
#include <stdexcept>

namespace ddrf
{
    namespace cufft
    {
        /*
         * CUFFT_ALLOC_FAILED
         */
        class bad_alloc : public std::exception
        {
            public:
                bad_alloc() noexcept = default;
                virtual ~bad_alloc() = default;

                auto operator=(const bad_alloc&) noexcept -> bad_alloc& {}

                virtual auto what() const noexcept -> const char*
                {
                    return "cuFFT failed to allocate GPU or CPU memory";
                }
        };

        /*
         * CUFFT_INVALID_PLAN
         * CUFFT_INVALID_TYPE
         * CUFFT_INVALID_VALUE
         * CUFFT_INVALID_SIZE
         * CUFFT_UNALIGNED_DATA
         * CUFFT_INCOMPLETE_PARAMETER_LIST
         * CUFFT_NO_WORKSPACE
         */
        class invalid_argument : public std::invalid_argument
        {
            public:
                using std::invalid_argument::invalid_argument;
                virtual ~invalid_argument() = default;
        };

        /*
         * CUFFT_INTERNAL_ERROR
         * CUFFT_EXEC_FAILED
         * CUFFT_SETUP_FAILED
         * CUFFT_INVALID_DEVICE
         * CUFFT_PARSE_ERROR
         * CUFFT_NOT_IMPLEMENTED
         * CUFFT_LICENSE_ERROR
         */
        class runtime_error : public std::runtime_error
        {
            public:
                using std::runtime_error::runtime_error;
                virtual ~runtime_error() = default;
        };
    }
}

#endif /* DDRF_CUFFT_EXCEPTION_H_ */
