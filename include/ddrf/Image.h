/*
 * Image.h
 *
 *  Created on: 05.11.2015
 *      Author: Jan Stephan
 *
 *      Image class that holds a pointer to the concrete image data
 *      The individual objects are usually created by the ImageHandler and its corresponding handler policies.
 */

#ifndef DDRF_IMAGE_H_
#define DDRF_IMAGE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddrf
{
	template <class Implementation>
	class Image : public Implementation
	{
		public:
			using value_type = typename Implementation::value_type;
			using pointer_type = typename Implementation::pointer_type;
			using size_type = typename Implementation::size_type;

		public:
			/*
			 * Constructs an empty (invalid) image.
			 */
			Image() noexcept
			: width_{0}, height_{0}, data_{nullptr}, valid_{false}
			{
			}

			/*
			 * Constructs an image with the given dimensions. If img_data is a nullptr the data_ member will
			 * be allocated. If img_data is not a nullptr the Image object will own the pointer that gets
			 * passed to it. In every case valid_ will be set to true after construction.
			 */
			Image(size_type img_width, size_type img_height,
					pointer_type&& img_data = nullptr)
			: width_{img_width}, height_{img_height}, data_{std::move(img_data)}, valid_{true}
			{
				if(data_ == nullptr)
						data_ = Implementation::make_ptr(width_, height_);
			}

			Image(const Image& other)
			: Implementation(other)
			, width_{other.width_}, height_{other.height_}, valid_{other.valid_}
			{
				if(other.data_ == nullptr)
					data_ = nullptr;
				else
				{
					data_ = Implementation::make_ptr(width_, height_);
					Implementation::copy(data_, other.data_, width_, height_);
				}
			}

			template <typename U>
			auto operator=(const Image<U>& rhs) -> Image<Implementation>&
			{
				width_ = rhs.width();
				height_ = rhs.height();
				valid_ = rhs.valid();

				if(rhs.container() == nullptr)
					data_ = nullptr;
				else
				{
					data_.reset(); // delete old content if any
					data_ = Implementation::make_ptr(width_, height_);
					Implementation::copy(data_, rhs.container(), width_, height_);
				}

				return *this;
			}

			Image(Image&& other) noexcept
			: Implementation(std::move(other))
			, width_{other.width_}, height_{other.height_}, data_{std::move(other.data_)}
			, valid_{other.valid_}
			{
				other.valid_ = false; // invalid after we moved its data
			}

			auto operator=(Image&& rhs) noexcept -> Image&
			{
				width_ = rhs.width_;
				height_ = rhs.height_;
				data_ = std::move(rhs.data_);
				valid_ = rhs.valid_;

				Implementation::operator=(std::move(rhs));

				rhs.valid_ = false;
				return *this;
			}

			auto width() const noexcept -> size_type
			{
				return width_;
			}

			auto height() const noexcept -> size_type
			{
				return height_;
			}

			/*
			 * returns a non-owning pointer to the data. Do not delete this pointer as the Image object will take
			 * care of the memory.
			 */
			auto data() const noexcept -> value_type*
			{
				return data_.get();
			}

			auto pitch() const noexcept -> size_type
			{
				return data_.pitch();
			}

			auto valid() const noexcept -> bool
			{
				return valid_;
			}

			/*
			 * return the underlying pointer
			 */
			auto container() const noexcept -> const pointer_type&
			{
				return data_;
			}

		private:
			size_type width_;
			size_type height_;
			pointer_type data_;
			bool valid_;
	};
}



#endif /* DDRF_IMAGE_H_ */
