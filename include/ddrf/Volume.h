#ifndef DDRF_VOLUME_H_
#define DDRF_VOLUME_H_

#include <stdexcept>
#include <utility>

#include "Image.h"

namespace ddrf
{
	template <class MemoryManager>
	class Volume : public MemoryManager
	{
		public:
			using value_type = typename MemoryManager::value_type;
			using pointer_type = typename MemoryManager::pointer_type_3D;
			using pointer_type_2D = typename MemoryManager::pointer_type_2D;
			using size_type = typename MemoryManager::size_type;

		public:
			Volume() noexcept
			: MemoryManager()
			, width_{0}, height_{0}, depth_{0}, data_{nullptr}, valid_{false}
			{}

			Volume(size_type w, size_type h, size_type d, pointer_type ptr = nullptr)
			: MemoryManager()
			, width_{w}, height_{h}, depth_{d}, data_{std::move(ptr)}, valid_{true}
			{
				if(data_ == nullptr)
					data_ = MemoryManager::make_ptr(width_, height_, depth_);
			}

			Volume(const Volume& other)
			: MemoryManager()
			, width_{other.width_}, height_{other.height_}, depth_{other.depth_}, valid_{other.valid_}
			{
				if(other.data_ == nullptr)
					data_ = nullptr;
				else
				{
					data_ = MemoryManager::make_ptr(width_, height_, depth_);
					MemoryManager::copy(data_, other.data_, width_, height_, depth_);
				}
			}

			template <typename U>
			auto operator=(const Volume<U>& rhs) -> Volume<MemoryManager>&
			{
				width_ = rhs.width();
				height_ = rhs.height();
				depth_ = rhs.depth();

				if(rhs.container() == nullptr)
					data_ = nullptr;
				else
				{
					data_ = MemoryManager::make_ptr(width_, height_, depth_);
					MemoryManager::copy(data_, rhs.container(), width_, height_, depth_);
				}

				return *this;
			}

			Volume(Volume&& other) noexcept
			: MemoryManager(std::move(other))
			, width_{other.width_}, height_{other.height_}, depth_{other.depth_}, data_{std::move(other.data_)}
			, valid_{other.valid_}
			{}

			auto operator=(Volume&& rhs) noexcept -> Volume&
			{
				MemoryManager::operator=(std::move(rhs));

				width_ = rhs.width_;
				height_ = rhs.height_;
				depth_ = rhs.depth_;
				data_ = std::move(rhs.data_);
				valid_ = rhs.valid_;

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

			auto depth() const noexcept -> size_type
			{
				return depth_;
			}

			auto pitch() const noexcept -> size_type
			{
				return data_.pitch();
			}

			auto data() const noexcept -> value_type*
			{
				return data_.get();
			}

			auto valid() const noexcept -> bool
			{
				return valid_;
			}

			auto container() const noexcept -> const pointer_type&
			{
				return data_;
			}

			auto operator[](size_type i) -> Image<MemoryManager>
			{
				using underlying = typename pointer_type_2D::underlying_type;
				if(i >= depth_)
					throw std::out_of_range{"Volume: invalid slice index"};

				auto sliceIdx = i * width_ * height_;
				auto slicePtr = pointer_type_2D(underlying(data_.get() + sliceIdx), width_ * sizeof(value_type), width_, height_);
				auto ptr = MemoryManager::make_ptr(width_, height_);
				MemoryManager::copy(ptr, slicePtr, width_, height_);
				return Image<MemoryManager>{width_, height_, std::move(ptr)};
			}

		private:
			size_type width_;
			size_type height_;
			size_type depth_;
			pointer_type data_;
			bool valid_;

	};
}



#endif /* VOLUME_H_ */
