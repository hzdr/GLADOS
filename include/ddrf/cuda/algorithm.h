#ifndef DDRF_CUDA_ALGORITHM_H_
#define DDRF_CUDA_ALGORITHM_H_

#include <cstddef>
#include <utility>

#include <ddrf/cuda/sync_policy.h>

namespace ddrf
{
    namespace cuda
    {
        template <class SyncPolicy, class D, class S>
        auto copy(SyncPolicy&& policy, D&& dest, const S&& src, std::size_t x) -> void
        {
            policy.copy(std::forward<D>(dest), std::forward<const S>(src), x);
        }

        template <class SyncPolicy, class D, class S>
        auto copy(SyncPolicy&& policy, D&& dest, const S&& src, std::size_t x, std::size_t y) -> void
        {
            policy.copy(std::forward<D>(dest), std::forward<const S>(src), x, y);
        }

        template <class SyncPolicy, class D, class S>
        auto copy(SyncPolicy&& policy, D&& dest, const S&& src, std::size_t x, std::size_t y, std::size_t z,
                    std::size_t dest_off_x = 0, std::size_t dest_off_y = 0, std::size_t dest_off_z = 0,
                    std::size_t src_off_x = 0, std::size_t src_off_y = 0, std::size_t src_off_z = 0)
        -> void
        {
            policy.copy(std::forward<D>(dest), std::forward<const S>(src), x, y, z, dest_off_x, dest_off_y, dest_off_z, src_off_x, src_off_y, src_off_z);
        }

        template <class D, class S, class... Parms>
        auto copy(D&& dest, S&& src, Parms... parms) -> void
        {
            copy(sync, std::forward<D>(dest), std::forward<S>(src), std::forward<Parms>(parms)...);
        }

        template <class SyncPolicy, class P, class... Dims>
        auto fill(SyncPolicy&& policy, P&& p, int value, Dims&&... dims) -> void
        {
            policy.fill(std::forward<P>(p), value, std::forward<Dims>(dims)...);
        }

        template <class SyncPolicy, class P>
        auto fill(SyncPolicy&& policy, P&& p, int value, std::size_t x, std::size_t y) -> void
        {
            policy.fill(std::forward<P>(p), value, x, y);
        }

        template <class SyncPolicy, class P>
        auto fill(SyncPolicy&& policy, P&& p, int value, std::size_t x, std::size_t y, std::size_t z) -> void
        {
            policy.fill(std::forward<P>(p), value, x, y, z);
        }

        template <class P, class... Dims>
        auto fill(P&& p, int value, Dims... dims) -> void
        {
            fill(sync, std::forward<P>(p), value, std::forward<Dims>(dims)...);
        }
    }
}



#endif /* ALGORITHM_H_ */
