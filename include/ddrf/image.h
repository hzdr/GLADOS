#ifndef DDRF_IMAGE_H_
#define DDRF_IMAGE_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <string>

#include <ddrf/cuda/bits/location.h>

namespace ddrf
{
    template <class T, class Allocator>
    class image
    {
        public:
            using value_type = T;
            using allocator_type = typename std::allocator_traits<Allocator>::allocator_type;
            using size_type = typename std::allocator_traits<Allocator>::size_type;
            using difference_type = typename std::allocator_traits<Allocator>::difference_type;
            using reference = value_type&;
            using const_reference = const value_type&;
            using pointer = typename std::allocator_traits<Allocator>::pointer;
            using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
            using iterator = pointer;
            using const_iterator = const iterator;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;

            explicit image(const Allocator& alloc = Allocator()) noexcept
            : alloc_{alloc}, width_{0}, height_{0}, data_{nullptr}
            {}

            image(size_type width, size_type height, const T& value = T(), const Allocator& alloc = Allocator())
            : alloc_{alloc}, width_{width}, height_{height}, data_{alloc_.allocate(width, height)}
            {
                alloc_.fill(data_, value, width_, height_);
            }

            explicit image(size_type width, size_type height)
            : image(width, height, T(), Allocator())
            {}

            image(pointer data, size_type width, size_type height, const Allocator& alloc = Allocator())
            : alloc_{alloc}, width_{width}, height_{height}, data_{data}
            {}

            image(const image& other, const Allocator& alloc)
            : alloc_{alloc}, width_{other.width_}, height_{other.height_}, data_{alloc_.allocate(width, height)}
            {}

            image(image&& other) noexcept
            : alloc_{std::move(other.alloc_)}
            , width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            image(image&& other, const Allocator& alloc) noexcept
            : alloc_{alloc}
            , width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            ~image()
            {
                alloc_.deallocate(data_, width_, height_);
            }

            auto operator=(const image& other) -> image&
            {
                width_ = other.width_;
                height_ = other.height_;
                alloc_.deallocate(data_);

                if(std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment())
                    alloc_ = other.alloc_;

                if(alloc_ != other.alloc_)
                    data_ = other.alloc_.allocate(width_, height_);
                else
                    data_ = alloc_.allocate(width_, height_);

                return *this;
            }

            auto operator=(image&& other) -> image&
            {
                width_ = other.width_;
                height_ = other.height_;

                if(std::allocator_traits<allocator_type>::propagate_on_container_move_assignment())
                    alloc_ = other.alloc_;
                else
                {
                    // TODO: Move element by element
                }

                return *this;
            }

            auto get_allocator() const -> allocator_type
            {
                return alloc_;
            }

            auto at(size_type x, size_type y) -> reference
            {
                if((x > width_) || (y > height_))
                    throw std::out_of_range{std::string{"Invalid pixel coordinates: "} + std::to_string(x) + std::string{"x"} + std::to_string(y)};

                return operator()(x, y);
            }

            auto at(size_type x, size_type y) const -> const_reference
            {
                if((x > width_) || (y > height_))
                    throw std::out_of_range{std::string{"Invalid pixel coordinates: "} + std::to_string(x) + std::string{"x"} + std::to_string(y)};

                return operator()(x, y);
            }

            auto operator()(size_type x, size_type y) -> reference
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Access is not supported for non-CPU memory");

                return data_[x + y * width_];
            }

            auto operator()(size_type x, size_type y) const -> const_reference
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Access is not supported for non-CPU memory");

                return data_[x + y * width];
            }

            auto data() noexcept -> T*
            {
                return data_;
            }

            auto data() const noexcept -> const T*
            {
                return data_;
            }

            auto begin() noexcept -> iterator
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Iteration is not supported for non-CPU memory");
                return data_;
            }

            auto begin() const noexcept -> const_iterator
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Iteration is not supported for non-CPU memory");
                return data_;
            }

            auto cbegin() const noexcept -> const_iterator
            {
                return begin();
            }

            auto end() noexcept -> iterator
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Iteration is not supported for non-CPU memory");
                return data_ + (width_ * height_);
            }

            auto end() const noexcept -> const_iterator
            {
                static_assert(allocator_type::memory_location == cuda::location::host, "Iteration is not supported for non-CPU memory");
                return data_ + (width_ * height_);
            }

            auto cend() const noexcept -> const_iterator
            {
                return end();
            }

            auto rbegin() noexcept -> reverse_iterator
            {
                return reverse_iterator(end() - 1);
            }

            auto rbegin() const noexcept -> const_reverse_iterator
            {
                return const_reverse_iterator(end() - 1);
            }

            auto crbegin() const noexcept -> const_reverse_iterator
            {
                return const_reverse_iterator(cend() - 1);
            }

            auto rend() noexcept -> reverse_iterator
            {
                return reverse_iterator(begin() - 1);
            }

            auto rend() const noexcept -> const_reverse_iterator
            {
                return const_reverse_iterator(begin() - 1);
            }

            auto crend() const noexcept -> const_reverse_iterator
            {
                return const_reverse_iterator(begin() - 1);
            }

            constexpr auto empty() noexcept -> bool
            {
                return (begin() == end()) ? true : false;
            }

            constexpr auto size() noexcept -> size_type
            {
                return width_ * height_;
            }

            auto fill(const T& value) -> void
            {
                alloc_.fill(pointer, value, width_, height_);
            }

            auto swap(image& other) -> void
            {
                std::swap(alloc_, other.alloc_);
                std::swap(width_, other.width_);
                std::swap(height_, other.height_);
                std::swap(data_, other.data_);
            }

        private:
            allocator_type alloc_;

            size_type width_;
            size_type height_;
            pointer data_;
    };
}



#endif /* IMAGE_H_ */
