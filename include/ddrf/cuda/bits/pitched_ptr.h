#ifndef PITCHED_PTR_H_
#define PITCHED_PTR_H_

#include <cstddef>

namespace ddrf
{
    namespace cuda
    {
        template <typename T>
        class pitched_ptr
        {
            public:
                explicit pitched_ptr(T* p, std::size_t ptr_pitch) noexcept : ptr_{p}, pitch_{ptr_pitch} {}
                explicit pitched_ptr(std::nullptr_t) noexcept : ptr_{nullptr}, pitch_{0} {}

                auto ptr() const noexcept -> T* { return ptr_; }
                auto pitch() const noexcept -> std::size_t { return pitch_; }

            private:
                T* ptr_;
                std::size_t pitch_;
        };

        template <class T>
        auto operator==(const pitched_ptr<T>& x, std::nullptr_t) -> bool
        {
            return x.ptr() == nullptr;
        }

        template <class T>
        auto operator==(std::nullptr_t, const pitched_ptr<T>& y) -> bool
        {
            return nullptr == y.ptr();
        }

        template <class T>
        auto operator!=(const pitched_ptr<T>&x, std::nullptr_t) -> bool
        {
            return !(x == nullptr);
        }

        template <class T>
        auto operator!=(std::nullptr_t, const pitched_ptr<T>& y) -> bool
        {
            return !(nullptr == y);
        }
    }
}

#endif /* PITCHED_PTR_H_ */
