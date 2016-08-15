#ifndef DDRF_IMAGE_H_
#define DDRF_IMAGE_H_

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <string>

#include <ddrf/memory.h>

namespace ddrf
{
    template <class T, class AllocatorT>
    class image_base
    {
        public:
            using value_type = T;
            using allocator_type = typename std::allocator_traits<AllocatorT>::allocator_type;
            using size_type = typename std::allocator_traits<AllocatorT>::size_type;
            using difference_type = typename std::allocator_traits<AllocatorT>::difference_type;
            using pointer = typename std::allocator_traits<AllocatorT>::pointer;
            using const_pointer = typename std::allocator_traits<AllocatorT>::const_pointer;

            explicit image_base(const AllocatorT& alloc = AllocatorT()) noexcept
            : alloc_{alloc}, width_{0}, height_{0}, data_{nullptr}
            {}

            image_base(size_type width, size_type height, const T& value = T(), const AllocatorT& alloc = AllocatorT())
            : alloc_{alloc}, width_{width}, height_{height}, data_{alloc_.allocate(width, height)}
            {
                alloc_.fill(data_, value, width_, height_);
            }

            explicit image_base(size_type width, size_type height)
            : image(width, height, T(), Allocator())
            {}

            image_base(pointer data, size_type width, size_type height, const AllocatorT& alloc = AllocatorT())
            : alloc_{alloc}, width_{width}, height_{height}, data_{data}
            {}

            image_base(const image_base& other, const AllocatorT& alloc)
            : alloc_{alloc}, width_{other.width_}, height_{other.height_}, data_{alloc_.allocate(width, height)}
            {}

            image_base(image_base&& other) noexcept
            : alloc_{std::move(other.alloc_)}
            , width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            image_base(image_base&& other, const AllocatorT& alloc) noexcept
            : alloc_{alloc}
            , width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
            {
                other.data_ = nullptr;
            }

            ~image_base()
            {
                alloc_.deallocate(data_, width_, height_);
            }

            auto operator=(const image_base& other) -> image_base&
            {
                width_ = other.width_;
                height_ = other.height_;
                alloc_.deallocate(data_);

                if(std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment())
                    alloc_ = other.alloc_;

                data_ = alloc_.allocate(width_, height_);

                // TODO: actual copy

                return *this;
            }

            auto operator=(image_base&& other) -> image_base&
            {
                width_ = other.width_;
                height_ = other.height_;
                alloc_.deallocate(data_);

                if(std::allocator_traits<allocator_type>::propagate_on_container_move_assignment())
                {
                    alloc_ = std::move(other.alloc_);
                    data _ = std::move(other.data_);
                }
                else
                {
                    // TODO: move element by element
                }

                return *this;
            }

            auto get_allocator() const noexcept -> allocator_type
            {
                return alloc_;
            }

            auto data() noexcept -> T*
            {
                return data_;
            }

            auto data() const noexcept -> const T*
            {
                return data_;
            }

            auto size() noexcept -> size_type
            {
                return width_ * height_;
            }

            auto width() noexcept -> size_type
            {
                return width_;
            }

            auto height() noexcept -> size_type
            {
                return height_;
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

        protected:
            size_type width_;
            size_type height_;
            pointer data_;

        private:
            allocator_type alloc_;
    };

    /*
     * Default image. Supports neither iteration nor direct access.
     */
    template <class T, class AllocatorT, class Enable = void>
    class image : public image_base<T, AllocatorT>
    {
        public:
            using value_type = typename base_class::value_type;
            using allocator_type = typename base_class::allocator_type;
            using size_type = typename base_class::size_type;
            using difference_type = typename base_class::difference_type;
            using pointer = typename base_class::pointer;
            using const_pointer = typename base_class::const_pointer;

            using image_base<T, AllocatorT>::image_base; // inherit constructor
    };

    /*
     * Host image specialization. Supports iteration and direct access.
     */
    template <class T, class AllocatorT>
    class image<T, AllocatorT, typename std::enable_if<AllocatorT::location == memory_location::host>::type> : public image_base<T, AllocatorT>
    {
        public:
            using value_type = typename base_class::value_type;
            using allocator_type = typename base_class::allocator_type;
            using size_type = typename base_class::size_type;
            using difference_type = typename base_class::difference_type;
            using reference = value_type&;
            using const_reference = const value_type&;
            using pointer = typename base_class::pointer;
            using const_pointer = typename base_class::const_pointer;
            using iterator = pointer;
            using const_iterator = const iterator;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        public:
            using image_base<T, AllocatorT>::image_base; // inherit constructor

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

            auto operator()(size_type x, size_type y) noexcept -> reference
            {
                return this->data_[x + y * width_];
            }

            auto operator()(size_type x, size_type y) const noexcept -> const_reference
            {
                return this->data_[x + y * width];
            }

            auto begin() noexcept -> iterator
            {
                return this->data_;
            }

            auto begin() const noexcept -> const_iterator
            {
                return this->data_;
            }

            auto cbegin() const noexcept -> const_iterator
            {
                return begin();
            }

            auto end() noexcept -> iterator
            {
                return this->data_ + (this->width_ * this->height_);
            }

            auto end() const noexcept -> const_iterator
            {
                return this->data_ + (this->width_ * this->height_);
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

            auto empty() noexcept -> bool
            {
                return (begin() == end()) ? true : false;
            }
    };
}



#endif /* IMAGE_H_ */
