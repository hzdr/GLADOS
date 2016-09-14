#ifndef DDRF_CUDA_ALGORITHM_H_
#define DDRF_CUDA_ALGORITHM_H_

#include <cstddef>
#include <utility>
#include <type_traits>

namespace ddrf
{
    namespace cuda
    {
        template <class SyncPolicy, class D, class S, class... Args>
        auto copy(SyncPolicy&& policy, D& dst, const S& src, Args... args) -> void
        {
            policy.copy(dst, src, std::forward<Args>(args)...);
        }

        /**
         * Note that fill() will apply value to the individual bytes of the data, not the elements
         */
        template <class SyncPolicy, class P, class... Dims>
        auto fill(SyncPolicy&& policy, P& p, int value, Dims&&... dims) -> void
        {
            policy.fill(p, value, std::forward<Dims>(dims)...);
        }
    }
}



#endif /* ALGORITHM_H_ */
