#ifndef DDRF_VOLUME_H_
#define DDRF_VOLUME_H_

#include <iterator>
#include <memory>

namespace ddrf
{
    template <class T, class Allocator>
    class volume
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

        public:
            explicit volume(const Allocator& alloc = Allocator()) noexcept
            : alloc_{alloc}, width_{0}, height_{0}, depth_{0}, data_{nullptr}
            {}

            volume(size_type width, size_type height, size_type depth, const T& value = T(), const Allocator& alloc = Allocator())
            : alloc_{alloc}, width_{width}, height_{height}, depth_{depth}, data_{alloc_.allocate(width_, height_, depth_)}
            {
                alloc_.fill(data_, value, width_, height_, depth_);
            }

            explicit volume(size_type width, size_type height, size_type depth)
            : volume(width, height, depth, T(), Allocator())
            {}

            volume(pointer data, size_type width, size_type height, size_type depth, const Allocator& alloc = Allocator())
            : alloc_{alloc}, width_{width}, height_{height}, depth_{depth}, data_{data}
            {}

            volume(const volume& other, const Allocator& alloc)
            : alloc_{alloc}, width_{other.width_}, height_{other.height_}, depth_{other.depth_}, data_{alloc_.allocate(width_, height_, depth_)}
            {}

            volume(volume&& other) noexcept
            : alloc_{std::move(other.alloc_)}, width_{other.width_}, height_{other.height_}, depth_{other.depth_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            volume(volume&& other, const Allocator& alloc) noexcept
            : alloc_{alloc}, width_{other.width_}, height_{other.height_}, depth_{other.depth_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            ~volume()
            {
                alloc_.deallocate(data_, width_, height_, depth_);
            }

            auto operator=(const volume& other) -> volume&
            {
                width_ = other.width_;
                height_ = other.height_;
                depth_ = other.depth_;
                alloc_.deallocate(data_);

                if(std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment())
                    alloc_ = other.alloc_;

                if(alloc_ != other.alloc_)
                    data_ = other.alloc_.allocate(width_, height_, depth_);
                else
                    data_ = alloc_.allocate(width_, height_, depth_);

                return *this;
            }

            auto operator=(volume&& other) -> volume&
            {
                width_ = other.width_;
                height_ = other.height_;
                depth_ = other.depth_;

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

            auto at(size_type x, size_type y, size_type z) -> reference
            {
                if((x > width_) || (y > height_) || (z > depth_))
                    throw std::out_of_range{std::string{"Invalid pixel coordinates: "}
                                            + std::to_string(x) + std::string{"x"} + std::to_string(y) + std::string{"x"} + std::to_string(z)};

                return operator()(x, y, z);
            }

            auto at(size_type x, size_type y, size_type z) const -> const_reference
            {
                if((x > width_) || (y > height_) || (z > depth_))
                    throw std::out_of_range{std::string{"Invalid pixel coordinates: "} + std::to_string(x) + std::string{"x"} + std::to_string(y)};

                return operator()(x, y, z);
            }

            auto operator()(size_type x, size_type y, size_type z) -> reference
            {
                static_assert(allocator_type::location == memory_location::host, "Access is not supported for non-CPU memory");

                return data_[x + y * width_ + z * height_];
            }

            auto operator()(size_type x, size_type y, size_type z) const -> const_reference
            {
                static_assert(allocator_type::location == memory_location::host, "Access is not supported for non-CPU memory");

                return data_[x + y * width + z * height_];
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
                static_assert(allocator_type::location == memory_location::host, "Iteration is not supported for non-CPU memory");
                return data_;
            }

            auto begin() const noexcept -> const_iterator
            {
                static_assert(allocator_type::location == memory_location::host, "Iteration is not supported for non-CPU memory");
                return data_;
            }

            auto cbegin() const noexcept -> const_iterator
            {
                return begin();
            }

            auto end() noexcept -> iterator
            {
                static_assert(allocator_type::location == memory_location::host, "Iteration is not supported for non-CPU memory");
                return data_ + (width_ * height_ * depth_);
            }

            auto end() const noexcept -> const_iterator
            {
                static_assert(allocator_type::location == memory_location::host, "Iteration is not supported for non-CPU memory");
                return data_ + (width_ * height_ * depth_);
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

            auto size() noexcept -> size_type
            {
                return width_ * height_ * depth_;
            }

            auto width() noexcept -> size_type
            {
                return width_;
            }

            auto height() noexcept -> size_type
            {
                return height_;
            }

            auto depth() noexcept -> size_type
            {
                return depth_;
            }

            auto fill(const T& value) -> void
            {
                alloc_.fill(pointer, value, width_, height_, depth_);
            }

            auto swap(image& other) -> void
            {
                std::swap(alloc_, other.alloc_);
                std::swap(width_, other.width_);
                std::swap(height_, other.height_);
                std::swap(depth_, other.depth_);
                std::swap(data_, other.data_);
            }

        private:
            allocator_type alloc_;

            size_type width_;
            size_type height_;
            size_type depth_
            pointer data_;
    };
}



#endif /* DDRF_VOLUME_H_ */
