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
#include <ddrf/cuda/bits/throw_error.h>

namespace ddrf
{
    namespace cuda
    {
        struct device_deleter { auto operator()(void* p) noexcept -> void { cudaFree(p); }};
        struct host_deleter { auto operator()(void* p) noexcept -> void { cudaFreeHost(p); }};

        template <class T, class Deleter, bool pitched, memory_location loc, bool pinned>
        class unique_ptr {};

        template <class T, class Deleter, memory_location loc>
        class unique_ptr<T, Deleter, true, loc, false> // As of now (CUDA 8.0) pinned host memory is never pitched
        {
            public:
                using pointer = pitched_ptr<T>;
                using element_type = T;
                using deleter_type = Deleter;

                static constexpr auto mem_location = loc;
                static constexpr auto pitched_memory = true;
                static constexpr auto pinned_memory = false;

                constexpr unique_ptr() noexcept : ptr_{nullptr}, pitch_{0u}, deleter_{}
                {}

                constexpr unique_ptr(std::nullptr_t) noexcept : ptr_{nullptr}, pitch_{0u}, deleter_{}
                {}

                /* FIXME: In C++17 the constructor will be templated to template<class U>(U ptr). Change the
                 * following constructor accordingly as soon as CUDA supports C++17
                 */
                explicit unique_ptr(pointer ptr) noexcept
                : ptr_{ptr.ptr()}, pitch_{ptr.pitch()}, deleter_{}
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
                : ptr_{ptr.ptr()}, pitch_{ptr.pitch()}, deleter_(d1)
                {}

                unique_ptr(pointer ptr,
                            typename std::remove_reference<deleter_type>::type&& d2) noexcept
                : ptr_{ptr.ptr()}, pitch_{ptr.pitch()}, deleter_(d2)
                {}

                unique_ptr(unique_ptr&& u)
                {
                    if(u != nullptr)
                    {
                        ptr_ = u.ptr_;
                        pitch_ = u.pitch_;
                        u.ptr_ = nullptr;
                        u.pitch_ = 0;

                        if(std::is_reference<deleter_type>::value)
                            deleter_ = u.deleter_;
                        else
                            deleter_ = std::move(u.deleter_);
                    }
                    else
                    {
                        ptr_ = nullptr;
                        pitch_ = 0;
                    }
                }

                ~unique_ptr()
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);
                }

                auto operator=(unique_ptr&& r) noexcept -> unique_ptr&
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);

                    if(r != nullptr)
                    {
                        ptr_ = r.ptr_;
                        pitch_ = r.pitch_;
                        r.ptr_ = nullptr;
                        r.pitch_ = 0;

                        if(std::is_reference<deleter_type>::value)
                            deleter_ = r.deleter_;
                        else
                            deleter_ = std::move(r.deleter_);
                    }
                    else
                    {
                        ptr_ = nullptr;
                        pitch_ = 0;
                    }

                    return *this;
                }

                auto operator=(std::nullptr_t) noexcept -> unique_ptr&
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);

                    ptr_ = nullptr;
                    pitch_ = 0;
                    return *this;
                }

                auto release() noexcept -> pointer
                {
                    auto ret = pointer{ptr_, pitch_};
                    ptr_ = nullptr;
                    return ret;
                }

                /*
                 * FIXME: Remove this method as soon as CUDA supports C++17
                 */
                auto reset(pointer ptr = pointer()) noexcept -> void
                {
                    auto old_ptr = ptr_;
                    ptr_ = ptr.ptr();
                    pitch_ = ptr.pitch();

                    if(old_ptr != nullptr)
                        deleter_(old_ptr);
                }

                /* FIXME: Change this to standard behaviour as soon as CUDA supports C++17. */
                template <class U>
                auto reset(U) noexcept -> void = delete;

                auto reset(std::nullptr_t) noexcept -> void
                {
                    auto old_ptr = ptr_;
                    ptr_ = nullptr;
                    pitch_ = 0;

                    if(old_ptr != nullptr)
                        deleter_(old_ptr);
                }

                auto swap(unique_ptr& other) noexcept -> void
                {
                    std::swap(ptr_, other.ptr_);
                    std::swap(pitch_, other.pitch_);
                    std::swap(deleter_, other.deleter_);
                }

                auto get() const noexcept -> element_type*
                {
                    return ptr_;
                }

                auto get_deleter() noexcept -> deleter_type&
                {
                    return deleter_;
                }

                auto get_deleter() const noexcept -> const deleter_type&
                {
                    return deleter_;
                }

                explicit operator bool() const noexcept
                {
                    return ptr_ != nullptr;
                }

                auto pitch() const noexcept -> std::size_t
                {
                    return pitch_;
                }

            private:
                element_type* ptr_;
                std::size_t pitch_;
                deleter_type deleter_;
        };

        template <class T, class Deleter, memory_location loc, bool pinned>
        class unique_ptr<T, Deleter, false, loc, pinned>
        {
            public:
                using pointer = T*;
                using element_type = T;
                using deleter_type = Deleter;

                static constexpr auto mem_location = loc;
                static constexpr auto pitched_memory = false;
                static constexpr auto pinned_memory = pinned;

                constexpr unique_ptr() noexcept : ptr_{nullptr}, deleter_{}
                {}

                constexpr unique_ptr(std::nullptr_t) noexcept : ptr_{nullptr}, deleter_{}
                {}

                /* FIXME: In C++17 the constructor will be templated to template<class U>(U ptr). Change the
                 * following constructor accordingly as soon as CUDA supports C++17
                 */
                explicit unique_ptr(pointer ptr) noexcept
                : ptr_{ptr}, deleter_{}
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
                : ptr_{ptr}, deleter_(d1)
                {}

                unique_ptr(pointer ptr,
                            typename std::remove_reference<deleter_type>::type&& d2) noexcept
                : ptr_{ptr}, deleter_(d2)
                {}

                unique_ptr(unique_ptr&& u)
                {
                    if(u != nullptr)
                    {
                        ptr_ = u.ptr_;
                        u.ptr_ = nullptr;
                    }
                    else
                        ptr_ = nullptr;

                    if(std::is_reference<deleter_type>::value)
                        deleter_ = u.deleter_;
                    else
                        deleter_ = std::move(u.deleter_);
                }

                ~unique_ptr()
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);
                }

                auto operator=(unique_ptr&& r) noexcept -> unique_ptr&
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);

                    if(r != nullptr)
                    {
                        ptr_ = r.ptr_;
                        if(std::is_reference<deleter_type>::value)
                            deleter_ = r.deleter_;
                        else
                            deleter_ = std::move(r.deleter_);

                        r.ptr_ = nullptr;
                    }
                    else
                        ptr_ = nullptr;

                    return *this;
                }

                auto operator=(std::nullptr_t) noexcept -> unique_ptr&
                {
                    if(ptr_ != nullptr)
                        deleter_(ptr_);

                    ptr_ = nullptr;
                    return *this;
                }

                auto release() noexcept -> pointer
                {
                    auto p = ptr_;
                    ptr_ = nullptr;
                    return p;
                }

                /*
                 * FIXME: Remove this method as soon as CUDA supports C++17
                 *
                 * Note that the following method has a different behaviour when
                 * compared to the STL's unique_ptr. The latter uses pointer() as a default
                 * argument.
                 */
                auto reset(pointer ptr = pointer()) noexcept -> void
                {
                    auto old_ptr = ptr;
                    ptr_ = ptr;

                    if(old_ptr != nullptr)
                        deleter_(ptr_);
                }

                /* FIXME: Change this to standard behaviour as soon as CUDA supports C++17. */
                template <class U>
                auto reset(U) noexcept -> void = delete;

                auto reset(std::nullptr_t) noexcept -> void
                {
                    auto old_ptr = ptr_;
                    ptr_ = nullptr;

                    if(old_ptr != nullptr)
                        deleter_(ptr_);
                }

                auto swap(unique_ptr& other) noexcept -> void
                {
                    std::swap(ptr_, other.ptr_);
                    std::swap(deleter_, other.deleter_);
                }

                auto get() const noexcept -> pointer
                {
                    return ptr_;
                }

                auto get_deleter() noexcept -> deleter_type&
                {
                    return deleter_;
                }

                auto get_deleter() const noexcept -> const deleter_type&
                {
                    return deleter_;
                }

                explicit operator bool() const noexcept
                {
                    return ptr_ != nullptr;
                }

                auto pitch() const noexcept -> std::size_t
                {
                    return 0;
                }

            private:
                pointer ptr_;
                deleter_type deleter_;
        };

        template <class T>
        using device_ptr = unique_ptr<T, device_deleter, false, memory_location::device, false>;

        template <class T>
        using pitched_device_ptr = unique_ptr<T, device_deleter, true, memory_location::device, false>;

        template <class T>
        using host_ptr = unique_ptr<T, std::default_delete<T[]>, false, memory_location::host, false>;

        template <class T>
        using pinned_host_ptr = unique_ptr<T, host_deleter, false, memory_location::host, true>;

        template <class T>
        auto make_unique_device(std::size_t n) -> device_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            auto err = cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T));
            if(err != cudaSuccess)
                detail::throw_error(err);
            return device_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_device(std::size_t x, std::size_t y) -> pitched_device_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            auto pitch = std::size_t{};
            auto err = cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, x * sizeof(T), y);
            if(err != cudaSuccess)
                detail::throw_error(err);
            return pitched_device_ptr<T>{pitched_ptr<T>{reinterpret_cast<T*>(ptr), pitch}};
        }

        template <class T>
        auto make_unique_device(std::size_t x, std::size_t y, std::size_t z) -> pitched_device_ptr<T>
        {
            auto extent = make_cudaExtent(x * sizeof(T), y, z);
            auto cuda_pitched_ptr = cudaPitchedPtr{};
            auto err = cudaMalloc3D(&cuda_pitched_ptr, extent);
            if(err != cudaSuccess)
                detail::throw_error(err);
            return pitched_device_ptr<T>{pitched_ptr<T>{reinterpret_cast<T*>(cuda_pitched_ptr.ptr), cuda_pitched_ptr.pitch}};
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
            auto err = cudaMallocHost(reinterpret_cast<void**>(&ptr), n * sizeof(T));
            if(err != cudaSuccess)
                detail::throw_error(err);
            return pinned_host_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_pinned_host(std::size_t x, std::size_t y) -> pinned_host_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            auto err = cudaMallocHost(reinterpret_cast<void**>(&ptr), x * y * sizeof(T));
            if(err != cudaSuccess)
                detail::throw_error(err);
            return pinned_host_ptr<T>{ptr};
        }

        template <class T>
        auto make_unique_pinned_host(std::size_t x, std::size_t y, std::size_t z) -> pinned_host_ptr<T>
        {
            auto ptr = static_cast<T*>(nullptr);
            auto err = cudaMallocHost(reinterpret_cast<void**>(&ptr), x * y * z * sizeof(T));
            if(err != cudaSuccess)
                detail::throw_error(err);
            return pinned_host_ptr<T>{ptr};
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator==(const unique_ptr<T1, D1, p1, l1, pn2>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y) -> bool
        {
            if(l1 != l2) return false; // same location?
            if(p1 != p2) return false; // same pointer type?
            if(pn1 != pn2) return false; // are both pinned / not pinned?
            return x.get() == y.get();
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator!=(const unique_ptr<T1, D1, p1, l1, pn1>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y) -> bool
        {
            return !(x == y);
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator<(const unique_ptr<T1, D1, p1, l1, pn1>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == memory_location::host) && (pn1 == pn2), bool>::type
        {
            return std::less<typename std::common_type
                                <typename unique_ptr<T1, D1, p1, l1, pn1>::pointer,
                                typename unique_ptr<T2, D2, p2, l2, pn2>::pointer>::type>()
                                (x.get(), y.get());
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator<=(const unique_ptr<T1, D1, p1, l1, pn1>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == memory_location::host) && (pn1 == pn2), bool>::type
        {
            return !(y < x);
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator>(const unique_ptr<T1, D1, p1, l1, pn1>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == memory_location::host) && (pn1 == pn2), bool>::type
        {
            return y < x;
        }

        template <class T1, class D1, bool p1, memory_location l1, bool pn1, class T2, class D2, bool p2, memory_location l2, bool pn2>
        auto operator>=(const unique_ptr<T1, D1, p1, l1, pn1>& x, const unique_ptr<T2, D2, p2, l2, pn2>& y)
        -> typename std::enable_if<(p1 == p2) && (l1 == l2) && (l1 == memory_location::host) && (pn1 == pn2), bool>::type
        {
            return !(x < y);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator==(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) noexcept -> bool
        {
            return !x;
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator==(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) noexcept -> bool
        {
            return !y;
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator!=(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) noexcept -> bool
        {
            return static_cast<bool>(x);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator!=(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) noexcept -> bool
        {
            return static_cast<bool>(y);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator<(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) -> bool
        {
            return std::less<typename unique_ptr<T, D, p, l, pn>::pointer>()(x.get(), nullptr);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator<(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) -> bool
        {
            return std::less<typename unique_ptr<T, D, p, l, pn>::pointer>()(nullptr, y.get());
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator<=(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) -> bool
        {
            return !(nullptr < x);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator<=(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) -> bool
        {
            return !(y < nullptr);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator>(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) -> bool
        {
            return nullptr < x;
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator>(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) -> bool
        {
            return y < nullptr;
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator>=(const unique_ptr<T, D, p, l, pn>& x, std::nullptr_t) -> bool
        {
            return !(x < nullptr);
        }

        template <class T, class D, bool p, memory_location l, bool pn>
        auto operator>=(std::nullptr_t, const unique_ptr<T, D, p, l, pn>& y) -> bool
        {
            return !(nullptr < y);
        }

        template <class T, class Deleter, bool pitched, memory_location loc, bool pinned>
        auto swap(unique_ptr<T, Deleter, pitched, loc, pinned>& lhs, unique_ptr<T, Deleter, pitched, loc, pinned>& rhs) noexcept -> void
        {
            lhs.swap(rhs);
        }
    }
}

#endif /* DDRF_CUDA_BITS_UNIQUE_PTR_H_ */
