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
	template <class MemoryManager>
	class Image : public MemoryManager
	{
		public:
			using value_type = typename MemoryManager::value_type;
			using pointer_type = typename MemoryManager::pointer_type_2D;
			using size_type = typename MemoryManager::size_type;

		public:
			Image() noexcept
			: width_{0}, height_{0}, data_{nullptr}, valid_{false}
			{
			}

			Image(size_type img_width, size_type img_height,
					pointer_type img_data = nullptr)
			: MemoryManager()
			, width_{img_width}, height_{img_height}, data_{std::move(img_data)}, valid_{true}
			{
				if(data_ == nullptr)
						data_ = MemoryManager::make_ptr(width_, height_);
			}

			Image(const Image& other)
			: MemoryManager(other)
			, width_{other.width_}, height_{other.height_}, valid_{other.valid_}
			{
				if(other.data_ == nullptr)
					data_ = nullptr;
				else
				{
					data_ = MemoryManager::make_ptr(width_, height_);
					MemoryManager::copy(data_, other.data_, width_, height_);
				}
			}

			template <typename U>
			auto operator=(const Image<U>& rhs) -> Image&
			{
				width_ = rhs.width();
				height_ = rhs.height();
				valid_ = rhs.valid();

				if(rhs.container() == nullptr)
					data_ = nullptr;
				else
				{
					data_.reset(); // delete old content if any
					data_ = MemoryManager::make_ptr(width_, height_);
					MemoryManager::copy(data_, rhs.container(), width_, height_);
				}

				return *this;
			}

			Image(Image&& other) noexcept
			: MemoryManager(std::move(other))
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

				MemoryManager::operator=(std::move(rhs));

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
