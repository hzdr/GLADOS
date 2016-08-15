#ifndef DDRF_CUDA_BITS_UNIQUE_PTR_H_
#define DDRF_CUDA_BITS_UNIQUE_PTR_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <ddrf/bits/memory_location.h>
#include <ddrf/cuda/exception.h>
#include <ddrf/cuda/bits/pitched_ptr.h>

namespace ddrf
{
    namespace cuda
    {
        struct device_deleter { auto operator()(void* p) noexcept -> void { cudaFree(p); }};
        struct host_deleter { auto operator()(void* p) noexcept -> void { cudaFreeHost(p); }};

        template <class T, class Deleter, bool pitched, detail::memory_location loc, bool pinned>
        class unique_ptr {};

        template <class T, class Deleter, detail::memory_location loc>
        class unique_ptr<T, Deleter, true, loc, false> // As of now (CUDA 7.5) pinned host memory is never pitched
        {
            public:
                using pointer = pitched_ptr<T>;
                using element_type = T;
                using deleter_type = Deleter;
                using memory_location = loc;

                static constexpr auto pitched_memory = true;
                static constexpr auto pinned_memory = false;

                constexpr unique_ptr() noexcept : ptr_{}, pitch_{0u}
                {}

                constexpr unique_ptr(std::nullptr_t) noexcept : ptr_{nullptr}, pitch_{0u}
                {}

                /* FIXME: In C++17 the constructor will be templated to template<class U>(U ptr). Change the
                 * following constructor accordingly as soon as CUDA supports C++17
                 */
                explicit unique_ptr(pointer ptr) noexcept
                : ptr_{ptr.ptr()}, pitch_{ptr.pitch()}
                {}

                /*
                 * The following 2 constructors construct a unique_ptr object as follows:
                 *  a)  If Deleter is non-reference type A, then the signatures are:
                 *          unique_ptr(pointer p, const A& d1);
                 *          unique_ptr(pointer p, A&& d2);
                 *  b)  If Deleter is an lvalue-reference type A&, then the signatures are:
                 *          unique_ptr(pointer p, A& d1);
                 *          unique_ptr(pointer p, A&& d2);
                 *  c)  If Deleter is an lvalue-reference type const A&, then the signatures are:
                 *          unique_ptr(pointer p, const A& d1);
                 *          unique_ptr(pointer p, const A&& d2);
                 */
                unique_ptr(pointer ptr,
                            typename std::conditional<std::is_reference<deleter_type>::value, deleter_type, const deleter_type&>::type d1) noexcept
                : ptr_{ptr.ptr(), d1}, pitch_{ptr.pitch()}
                {}

                unique_ptr(pointer ptr,
                            typename std::remove_reference<deleter_type>::type&& d2) noexcept
                : ptr_{ptr.ptr(), d2}, pitch_{ptr.pitch()}
                {}

                unique_ptr(unique_ptr&& u)
                {
                    ptr_ = std::move(u.ptr_);
                    pitch_ = u.pitch_;
                }

                ~unique_ptr() = default;

                auto operator=(unique_ptr&& r) noexcept -> unique_ptr&
                {
                    ptr_ = std::move(r.ptr_);
                    pitch_ = r.pitch_;
                    return *this;
                }

                auto operator=(std::nullptr_t) noexcept -> unique_ptr&
                {
                    ptr_ = nullptr;
                    return *this;
                }

                auto release() noexcept -> pointer
                {
                    return ptr_.release();
                }

                /*
                 * FIXME: Remove this method as soon as CUDA supports C++17
                 *
                 * Note that the following method has a slightly different behaviour when
                 * compared to the STL's unique_ptr. The latter uses pointer() as a default
                 * argument while this version does not.
                 */
                auto reset(pointer ptr) noexcept -> void
                {
                    ptr_.reset(ptr.ptr());
                    pitch_ = ptr.pitch();
                }

                /* FIXME: Change this to standard behaviour as soon as CUDA supports C++17. */
                template <class U>
                auto reset(U) noexcept -> void = delete;

                auto reset(std::nullptr_t p) noexcept -> void
                {
                    ptr_.reset(p);
                    pitch_ = 0;
                }

                auto swap(unique_ptr& other) noexcept -> void
                {
                    ptr_.swap(other.ptr_);
                    auto tmp = pitch_;
                    pitch_ = other.pitch_;
                    other.pitch_ = tmp;
                }

                auto get() const noexcept -> pointer
                {
                    return ptr_.get();
                }

                auto get_deleter() noexcept -> deleter_type&
                {
                    return ptr_.get_deleter();
                }

                auto get_deleter() const noexcept -> const deleter_type&
                {
                    return ptr_.get_deleter();
                }

                explicit operator bool() const noexcept
                {
                    return get() != nullptr;
                }

                auto pitch() const noexcept -> std::size_t
                {
                    return pitch_;
                }

            private:
                std::unique_ptr<element_type[], deleter_type> ptr_;
                std::size_t pitch_;
        };

        template <class T, class Deleter, detail::memory_location loc, bool pinned>
        class unique_ptr<T, Deleter, false, loc, pinned>
        {
            public:
                using pointer = T*;
                using element_type = T;
                using deleter_type = Deleter;
                using memory_location = loc;

                static constexpr auto pitched_memory = false;
                static constexpr auto pinned_memory = pinned;

                constexpr unique_ptr() noexcept : ptr_{}
                {}

                constexpr unique_ptr(std::nullptr_t) noexcept : ptr_{nullptr}
                {}

                /* FIXME: In C++17 the constructor will be templated to template<class U>(U ptr). Change the
                 * following constructor accordingly as soon as CUDA supports C++17
                 */
                explicit unique_ptr(pointer ptr) noexcept
                : ptr_{ptr}
                {}

                /*
                 * The following 2 constructors construct a unique_ptr object as follows:
                 *  a)  If Deleter is non-reference type A, then the signatures are:
                 *          unique_ptr(pointer p, const A& d1);
                 *          unique_ptr(pointer p, A&& d2);
                 *  b)  If Deleter is an lvalue-reference type A&, then the signatures are:
                 *          unique_ptr(pointer p, A& d1);
                 *          unique_ptr(pointer p, A&& d2);
                 *  c)  If Deleter is an lvalue-reference type const A&, then the signatures are:
                 *          unique_ptr(pointer p, const A& d1);
                 *          unique_ptr(pointer p, const A&& d2);
                 */
                unique_ptr(pointer ptr,
                            typename std::conditional<std::is_reference<deleter_type>::value, deleter_type, const deleter_type&>::type d1) noexcept
                : ptr_{ptr, d1}
                {}

                unique_ptr(pointer ptr,
                            typename std::remove_reference<deleter_type>::type&& d2) noexcept
                : ptr_{ptr, d2}
                {}

                unique_ptr(unique_ptr&& u)
                : ptr_{std::move(u.ptr_)}
                {}

                ~unique_ptr() = default;

                auto operator=(unique_ptr&& r) noexcept -> unique_ptr&
                {
                    ptr_ = std::move(r.ptr_);
                    return *this;
                }

                auto operator=(std::nullptr_t) noexcept -> unique_ptr&
                {
                    ptr_ = nullptr;
                    return *this;
                }

                auto release() noexcept -> pointer
                {
                    return ptr_.release();
                }

                /*
                 * FIXME: Remove this method as soon as CUDA supports C++17
                 *
                 * Note that the following method has a different behaviour when
                 * compared to the STL's unique_ptr. The latter uses pointer() as a default
                 * argument.
                 */
                auto reset(pointer ptr) noexcept -> void
                {
                    ptr_.reset(ptr);
                }

                /* FIXME: Change this to standard behaviour as soon as CUDA supports C++17. */
                template <class U>
                auto reset(U) noexcept -> void = delete;

                auto reset(std::nullptr_t p) noexcept -> void
                {
                    ptr_.reset(p);
                }

                auto swap(unique_ptr& other) noexcept -> void
                {
                    ptr_.swap(other.ptr_);
                }

                auto get() const noexcept -> pointer
                {
                    return ptr_.get();
                }

                auto get_deleter() noexcept -> deleter_type&
                {
                    return ptr_.get_deleter();
                }

                auto get_deleter() const noexcept -> const deleter_type&
                {
                    return ptr_.get_deleter();
                }

                explicit operator bool() const noexcept
                {
                    return get() != nullptr;
                }

                auto pitch() const noexcept -> std::size_t
                {
                    return 0;
                }

            private:
                std::unique_ptr<element_type[], deleter_type> ptr_;
        };

        template <class T>
        using device_ptr = unique_ptr<T, device_deleter, false, detail::memory_location::device, false>;

        template <class T>
        using pitched_device_ptr = unique_ptr<T, device_deleter, true, detail::memory_location::device, false>;

        template <class T>
        using host_ptr = unique_ptr<T, std::default_delete<T[]>, false, detail::memory_location::host, false>;

        template <class T>
        using pinned_host_ptr = unique_ptr<T, host_deleter, false, detail::memory_location::host, true>;

        template <class T>
        auto make_unique_device(std::size_t n) -> device_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            if(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return device_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_device(std::size_t x, std::size_t y) -> pitched_device_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            auto pitch = std::size_t{};
            if(cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, x * sizeof(T), y) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return pitched_device_ptr<T>{ptr, pitch};
        }

        template <class T>
        auto make_unique_device(std::size_t x, std::size_t y, std::size_t z) -> pitched_device_ptr<T>
        {
            auto extent = make_cudaExtent(x * sizeof(T), y, z);
            auto pitched_ptr = cudaPitchedPtr{};
            if(cudaMalloc3D(&pitched_ptr, extent) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return pitched_device_ptr<T>{pitched_ptr.ptr, pitched_ptr.pitch};
        }

        template <class T>
        auto make_unique_host(std::size_t n) -> host_ptr<T>
        {
            return host_ptr<T>{new T[n]};
        }

        template <class T>
        auto make_unique_host(std::size_t x, std::size_t y) -> host_ptr<T>
        {
            return host_ptr<T>{new T[x * y]};
        }

        template <class T>
        auto make_unique_host(std::size_t x, std::size_t y, std::size_t z) -> host_ptr<T>
        {
            return host_ptr<T>{new T[x * y * z]};
        }

        template <class T>
        auto make_unique_pinned_host(std::size_t n) -> pinned_host_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            if(cudaMallocHost(reinterpret_cast<void**>(&ptr), n * sizeof(T)) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return pinned_host_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_pinned_host(std::size_t x, std::size_t y) -> pinned_host_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            if(cudaMallocHost(reinterpret_cast<void**>(&ptr), x * y * sizeof(T)) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return pinned_host_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_pinned_host(std::size_t x, std::size_t y, std::size_t z) -> pinned_host_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            if(cudaMallocHost(reinterpret_cast<void**>(&ptr), x * y * z * sizeof(T)) == cudaErrorMemoryAllocation)
                throw bad_alloc();
            return pinned_host_ptr<T>{ptr};
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator==(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y) -> bool
        {
            if(l1 != l2) return false; // same location?
            if(p1 != p2) return false; // same pointer type?
            return x.get() == y.get();
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator!=(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y) -> bool
        {
            return !(x == y);
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator<(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == location::host), bool>::type
        {
            return std::less<typename std::common_type
                                <typename unique_ptr<T1, D1, p1, l1>::pointer,
                                typename unique_ptr<T2, D2, p2, l2>::pointer>::type>()
                                (x.get(), y.get());
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator<=(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == location::host), bool>::type
        {
            return !(y < x);
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator>(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == location::host), bool>::type
        {
            return y < x;
        }

        template <class T1, class D1, bool p1, location l1, class T2, class D2, bool p2, location l2>
        auto operator>=(const unique_ptr<T1, D1, p1, l1>& x, const unique_ptr<T2, D2, p2, l2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == location::host), bool>::type
        {
            return !(x < y);
        }

        template <class T, class D, bool p, location l>
        auto operator==(const unique_ptr<T, D, p, l>& x, std::nullptr_t) noexcept -> bool
        {
            return !x;
        }

        template <class T, class D, bool p, location l>
        auto operator==(std::nullptr_t, const unique_ptr<T, D, p, l>& y) noexcept -> bool
        {
            return !y;
        }

        template <class T, class D, bool p, location l>
        auto operator!=(const unique_ptr<T, D, p, l>& x, std::nullptr_t) noexcept -> bool
        {
            return static_cast<bool>(x);
        }

        template <class T, class D, bool p, location l>
        auto operator!=(std::nullptr_t, const unique_ptr<T, D, p, l>& y) noexcept -> bool
        {
            return static_cast<bool>(y);
        }

        template <class T, class D, bool p, location l>
        auto operator<(const unique_ptr<T, D, p, l>& x, std::nullptr_t) -> bool
        {
            return std::less<typename unique_ptr<T, D, p, l>::pointer>()(x.get(), nullptr);
        }

        template <class T, class D, bool p, location l>
        auto operator<(std::nullptr_t, const unique_ptr<T, D, p, l>& y) -> bool
        {
            return std::less<typename unique_ptr<T, D, p, l>::pointer>()(nullptr, y.get());
        }

        template <class T, class D, bool p, location l>
        auto operator<=(const unique_ptr<T, D, p, l>& x, std::nullptr_t) -> bool
        {
            return !(nullptr < x);
        }

        template <class T, class D, bool p, location l>
        auto operator<=(std::nullptr_t, const unique_ptr<T, D, p, l>& y) -> bool
        {
            return !(y < nullptr);
        }

        template <class T, class D, bool p, location l>
        auto operator>(const unique_ptr<T, D, p, l>& x, std::nullptr_t) -> bool
        {
            return nullptr < x;
        }

        template <class T, class D, bool p, location l>
        auto operator>(std::nullptr_t, const unique_ptr<T, D, p, l>& y) -> bool
        {
            return y < nullptr;
        }

        template <class T, class D, bool p, location l>
        auto operator>=(const unique_ptr<T, D, p, l>& x, std::nullptr_t) -> bool
        {
            return !(x < nullptr);
        }

        template <class T, class D, bool p, location l>
        auto operator>=(std::nullptr_t, const unique_ptr<T, D, p, l>& y) -> bool
        {
            return !(nullptr < y);
        }

        template <class T, class Deleter, bool pitched, detail::memory_location loc>
        auto swap(unique_ptr<T, Deleter, pitched, loc>& lhs, unique_ptr<T, Deleter, pitched, loc>& rhs) noexcept -> void
        {
            lhs.swap(rhs);
        }
    }
}

#endif /* DDRF_CUDA_BITS_UNIQUE_PTR_H_ */
