#ifndef DDRF_MEMORY_H_
#define DDRF_MEMORY_H_

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

/**
 * ddRF provides two types of pointers: A simple ("one-dimensional") one and a pitched ("multi-dimensional") one. Implementations for
 * different hardware have to provide CopyPolicy and Ptr for the underlying operations.
 */

namespace ddrf
{
	template <class T, class CopyPolicy, class Ptr>
	class base_ptr : public CopyPolicy
	{
		public:
			using pointer = typename Ptr::pointer;
			using element_type = typename Ptr::element_type;
			using deleter_type = typename Ptr::deleter_type;
			using underlying_type = Ptr;

		public:
			constexpr base_ptr() noexcept
			: ptr_{}
			{}

			constexpr base_ptr(std::nullptr_t p) noexcept
			: ptr_{p}
			{}

			base_ptr(Ptr ptr) noexcept
			: ptr_{std::move(ptr)}
			{}

			base_ptr(base_ptr&& other) noexcept
			: ptr_{std::move(other.ptr_)}
			{}

			~base_ptr() = default;

			inline auto operator=(base_ptr&& r) noexcept -> base_ptr&
			{
				ptr_ = std::move(r.ptr_);
				return *this;
			}

			inline auto release() noexcept -> pointer { return ptr_.release(); }
			inline auto reset(pointer ptr = pointer()) noexcept -> void { ptr_.reset(ptr); }
			template <class U> auto reset(U) noexcept -> void = delete;
			inline auto reset(std::nullptr_t p) noexcept -> void { ptr_.reset(p); }
			inline auto swap(base_ptr& other) noexcept -> void { ptr_.swap(other.ptr_); }
			inline auto get() const noexcept -> pointer { return ptr_.get(); }
			inline auto get_deleter() noexcept -> deleter_type { return ptr_.get_deleter(); }
			inline auto get_deleter() const noexcept -> const deleter_type { return ptr_.get_deleter(); }
			explicit inline operator bool() const noexcept { return ptr_.operator bool(); }
			inline auto operator[](std::size_t i) const -> element_type& { return ptr_[i]; }

			// this is nasty but currently I see no other way to access the ptr_ of different instances
			inline const Ptr& get_underlying() const noexcept { return ptr_; }

		protected:
			template <class Dest, class Src, class... Args>
			inline auto copy(Dest& dest, const Src& src, Args... args) -> void
			{
				CopyPolicy::copy(dest, src, args...);
			}

		private:
			base_ptr(const base_ptr&) = delete;
			auto operator=(const base_ptr&) -> base_ptr& = delete;

		protected:
			Ptr ptr_;

	};

	template <class T1, class C1, class Ptr1, class T2, class C2, class Ptr2>
	inline auto operator==(const base_ptr<T1, C1, Ptr1>& x, const base_ptr<T2, C2, Ptr2>& y) noexcept -> bool
	{
		return x.get() == y.get();
	}

	template <class T1, class C1, class Ptr1, class T2, class C2, class Ptr2>
	inline auto operator!=(const base_ptr<T1, C1, Ptr1>& x, const base_ptr<T2, C2, Ptr2>& y) noexcept -> bool
	{
		return x.get() != y.get();
	}

	template <class T, class C, class Ptr>
	inline auto operator==(const base_ptr<T, C, Ptr>& x, std::nullptr_t) noexcept -> bool
	{
		return !x;
	}

	template <class T, class C, class Ptr>
	inline auto operator==(std::nullptr_t, const base_ptr<T, C, Ptr>& x) noexcept -> bool
	{
		return !x;
	}

	template <class T, class C, class Ptr>
	inline auto operator!=(const base_ptr<T, C, Ptr>& x, std::nullptr_t) noexcept -> bool
	{
		return static_cast<bool>(x);
	}

	template <class T, class C, class Ptr>
	inline auto operator!=(std::nullptr_t, const base_ptr<T, C, Ptr>& x) noexcept -> bool
	{
		return static_cast<bool>(x);
	}

	template <class T, class CopyPolicy, class Ptr>
	class ptr : public base_ptr<T, CopyPolicy, Ptr>
	{
		private:
			using base = base_ptr<T, CopyPolicy, Ptr>;

		public:
			using pointer = typename base::pointer;
			using element_type = typename base::element_type;
			using deleter_type = typename base::deleter_type;
			using underlying_type = typename base::underlying_type;
			static constexpr auto has_pitch = false;

		public:
			constexpr ptr() noexcept
			: base()
			, size_{0u}
			{}

			constexpr ptr(std::nullptr_t p) noexcept
			: base(p)
			, size_{0u}
			{}

			ptr(Ptr p, std::size_t s) noexcept
			: base(std::move(p))
			, size_{s}
			{}

			ptr(ptr&& other) noexcept
			: size_{other.size_}
			, base(std::move(other))
			{}

			~ptr() = default;

			inline auto operator=(ptr&& r) noexcept -> ptr&
			{
				size_ = r.size_;
				base::operator=(std::move(r));
				return *this;
			}

			template <class OtherPtr>
			inline auto operator=(const ptr<T, CopyPolicy, OtherPtr>& rhs) -> ptr&
			{
				base::copy(*this, rhs, size_);
				return *this;
			}

			inline auto size() const noexcept -> std::size_t { return size_; }

		private:
			ptr(const ptr&) = delete;

		private:
			std::size_t size_;
	};

	template <class T, class CopyPolicy, class is3D, class Ptr>
	class pitched_ptr : public base_ptr<T, CopyPolicy, Ptr>
	{
		private:
			using base = base_ptr<T, CopyPolicy, Ptr>;

		public:
			using pointer = typename base::pointer;
			using element_type = typename base::element_type;
			using deleter_type = typename base::deleter_type;
			using underlying_type = typename base::underlying_type;

			static constexpr auto has_pitch = true;
			static constexpr auto is3DPtr = is3D::value;

		public:
			constexpr pitched_ptr() noexcept
			: base()
			, pitch_{0u}, width_{0u}, height_{0u}, depth_{0u}
			{}

			constexpr pitched_ptr(std::nullptr_t p) noexcept
			: base(p)
			, pitch_{0u}, width_{0u}, height_{0u}, depth_{0u}
			{}

			pitched_ptr(Ptr ptr, std::size_t p, std::uint32_t w, std::uint32_t h, std::uint32_t d = 0) noexcept
			: base(std::move(ptr))
			, pitch_{p}, width_{w}, height_{h}, depth_{d}
			{}

			pitched_ptr(pitched_ptr&& other) noexcept
			: base(std::move(other))
			, pitch_{other.pitch_}, width_{other.width_}, height_{other.height_}, depth_{other.depth_}
			{}

			~pitched_ptr() = default;

			auto operator=(pitched_ptr&& r) noexcept -> pitched_ptr&
			{
				pitch_ = r.pitch_;
				width_ = r.width_;
				height_ = r.height_;
				depth_ = r.depth_;
				base::operator=(std::move(r));

				return *this;
			}

			inline auto pitch() const noexcept -> std::size_t { return pitch_; }
			inline auto width() const noexcept -> std::size_t {	return width_; }
			inline auto height() const noexcept -> std::size_t { return height_; }
			inline auto depth() const noexcept -> std::size_t { return depth_; }

			// FIXME: enable_if around is3D
			template <class OtherPtr>
			inline auto operator=(const pitched_ptr<T, CopyPolicy, std::true_type, OtherPtr>& rhs) -> pitched_ptr&
			{
				base::copy(*this, rhs, width_, height_, depth_);
				return *this;
			}

			template <class OtherPtr>
			inline auto operator=(const pitched_ptr<T, CopyPolicy, std::false_type, OtherPtr>& rhs) -> pitched_ptr&
			{
				base::copy(*this, rhs, width_, height_);
				return *this;
			}

		private:
			pitched_ptr(const pitched_ptr&) = delete;

		private:
			std::size_t pitch_;
			std::size_t width_;
			std::size_t height_;
			std::size_t depth_;
	};
}



#endif /* DDRF_MEMORY_H_ */
