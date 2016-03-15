#ifndef CUDA_MEMORY_H_
#define CUDA_MEMORY_H_

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddrf/Memory.h>

#include "Check.h"

namespace ddrf
{
	namespace cuda
	{
		namespace detail
		{
			// no check() here as deleters are required to never throw exceptions
			struct device_deleter {	auto operator()(void* p) -> void { cudaFree(p); }};
			struct host_deleter	{ auto operator()(void* p) -> void { cudaFreeHost(p); }};

			enum class Target { Device, Host };

			template <Target Src, Target Dest> struct Direction {};
			template <> struct Direction<Target::Host, Target::Device> { static constexpr auto value = cudaMemcpyHostToDevice; };
			template <> struct Direction<Target::Device, Target::Host> { static constexpr auto value = cudaMemcpyDeviceToHost; };
			template <> struct Direction<Target::Device, Target::Device> { static constexpr auto value = cudaMemcpyDeviceToDevice; };
			template <> struct Direction<Target::Host, Target::Host> { static constexpr auto value = cudaMemcpyHostToHost; };

			template <class T, class Deleter, Target t>
			class unique_ptr
			{
				public:
					using pointer = T*;
					using element_type = T;
					using deleter_type = Deleter;
					static constexpr auto target = Target{t};

				public:
					constexpr unique_ptr() noexcept
					: ptr_{}
					{}

					constexpr unique_ptr(std::nullptr_t) noexcept
					: ptr_{nullptr}
					{}

					unique_ptr(pointer p) noexcept
					: ptr_{p}
					{}

					unique_ptr(unique_ptr&& other) noexcept
					: ptr_{std::move(other.ptr_)}
					{}

					inline auto operator=(unique_ptr&& r) noexcept -> unique_ptr&
					{
						ptr_ = std::move(r.ptr_);
						return *this;
					}

					inline auto operator=(std::nullptr_t) noexcept -> unique_ptr&
					{
						ptr_ = nullptr;
						return *this;
					}

					inline auto release() noexcept -> pointer {	return ptr_.release(); }
					inline auto reset(pointer ptr = pointer()) noexcept -> void { ptr_.reset(ptr); }

					template <class U>
					auto reset(U) noexcept -> void = delete;

					inline auto reset(std::nullptr_t) noexcept -> void { ptr_.reset(); }
					inline auto swap(unique_ptr& other) noexcept -> void { ptr_.swap(other.ptr_); }
					inline auto get() const noexcept -> pointer { return ptr_.get(); }
					inline auto get_deleter() noexcept -> Deleter& { return ptr_.get_deleter(); }
					inline auto get_deleter() const noexcept -> const Deleter& { return ptr_.get_deleter(); }
					explicit inline operator bool() const noexcept { return ptr_.operator bool(); }
					inline auto operator[](std::size_t i) const -> T& { return ptr_[i]; }

				private:
					unique_ptr(const unique_ptr&) = delete;
					auto operator=(const unique_ptr&) -> unique_ptr& = delete;

				private:
					std::unique_ptr<T[], Deleter> ptr_;
			};
			template <class T> using unique_device_ptr = unique_ptr<T, device_deleter, Target::Device>;
			template <class T> using unique_host_ptr = unique_ptr<T, host_deleter, Target::Host>;

			template <class T1, class D1, Target t1, class T2, class D2, Target t2>
			inline auto operator==(const unique_ptr<T1, D1, t2>& x, const unique_ptr<T2, D2, t2>& y) noexcept -> bool
			{
				return x.get() == y.get();
			}

			template <class T1, class D1, Target t1, class T2, class D2, Target t2>
			inline auto operator!=(const unique_ptr<T1, D1, t2>& x, const unique_ptr<T2, D2, t2>& y) noexcept -> bool
			{
				return x.get() != y.get();
			}

			template <class T, class D, Target t>
			inline auto operator==(const unique_ptr<T, D, t>& x, std::nullptr_t) noexcept -> bool
			{
				return !x;
			}

			template <class T, class D, Target t>
			inline auto operator==(std::nullptr_t, const unique_ptr<T, D, t>&x) noexcept -> bool
			{
				return !x;
			}

			template <class T, class D, Target t>
			inline auto operator!=(const unique_ptr<T, D, t>& x, std::nullptr_t) noexcept -> bool
			{
				return static_cast<bool>(x);
			}

			template <class T, class D, Target t>
			inline auto operator!=(std::nullptr_t, const unique_ptr<T, D, t>&x) noexcept -> bool
			{
				return static_cast<bool>(x);
			}
		}

		class sync_copy_policy
		{
			protected:
				~sync_copy_policy() = default;

				/* 1D copies*/
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t size) const -> void
				{
					check(cudaMemcpy(dest.get(), src.get(), size,
							detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value));
				}

				/* 2D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height) const -> void
				{
					check(cudaMemcpy2D(dest.get(), dest.pitch(),
										src.get(), src.pitch(),
										width * sizeof(typename Src::element_type), height,
										detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value));
				}

				/* 3D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height, std::size_t depth) const -> void
				{
					auto parms = cudaMemcpy3DParms{0};
					// using src.pitch() instead of width because cudaMemcpy3D interprets the pointer's elements as unsigned char
					parms.srcPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(src.get()), src.pitch(), src.pitch(), height);
					parms.dstPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(dest.get()), dest.pitch(), dest.pitch(), height);
					parms.extent = make_cudaExtent(width, height, depth);
					parms.kind = detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value;

					check(cudaMemcpy3D(&parms));
				}
		};

		class async_copy_policy
		{
			public:
				inline auto set_stream(cudaStream_t s) noexcept -> void
				{
					stream_ = s;
				}

				inline auto stream() const noexcept -> cudaStream_t
				{
					return stream_;
				}

			protected:
				async_copy_policy()
				: stream_{nullptr}
				{}

				~async_copy_policy() = default;

				/* 1D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t size) const -> void
				{
					check(cudaMemcpyAsync(dest.get(), src.get(), size,
							detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value, stream_));
				}

				/* 2D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height) const -> void
				{
					check(cudaMemcpy2DAsync(dest.get(), dest.pitch(),
											src.get(), src.pitch(),
											width * sizeof(typename Src::element_type), height,
											detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value, stream_));
				}

				/* 3D copies */
				template <class Dest, class Src>
				inline auto copy(Dest& dest, const Src& src, std::size_t width, std::size_t height, std::size_t depth) const -> void
				{
					auto parms = cudaMemcpy3DParms{0};
					parms.srcPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(src.get()), src.pitch(), src.pitch(), height);
					parms.dstPtr = make_cudaPitchedPtr(reinterpret_cast<unsigned char*>(dest.get()), dest.pitch(), dest.pitch(), height);
					parms.extent = make_cudaExtent(width, height, depth);
					parms.kind = detail::Direction<Src::underlying_type::target, Dest::underlying_type::target>::value;

					check(cudaMemcpy3DAsync(&parms, stream_));
				}

			private:
				cudaStream_t stream_;
		};

		template <class T, class CopyPolicy> using device_ptr = ddrf::ptr<T, CopyPolicy, detail::unique_device_ptr<T>>;
		template <class T, class CopyPolicy> using host_ptr = ddrf::ptr<T, CopyPolicy, detail::unique_host_ptr<T>>;

		template <class T, class CopyPolicy, class is3D> using pitched_device_ptr = ddrf::pitched_ptr<T, CopyPolicy, is3D, detail::unique_device_ptr<T>>;
		template <class T, class CopyPolicy, class is3D> using pitched_host_ptr = ddrf::pitched_ptr<T, CopyPolicy, is3D, detail::unique_host_ptr<T>>;

		/*
		 * Array types with unknown bounds
		 */
		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_device_ptr(std::size_t length) -> device_ptr<T, CopyPolicy>
		{
			auto p = static_cast<T*>(nullptr);
			auto size = length * sizeof(T);
			check(cudaMalloc(&p, size));
			return device_ptr<T, CopyPolicy>(detail::unique_device_ptr<T>(p), size);
		}

		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_host_ptr(std::size_t length) -> host_ptr<T, CopyPolicy>
		{
			auto p = static_cast<T*>(nullptr);
			auto size = length * sizeof(T);
			check(cudaMallocHost(&p, size));
			return host_ptr<T, CopyPolicy>(detail::unique_host_ptr<T>(p), size);
		}

		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_device_ptr(std::size_t width, std::size_t height) -> pitched_device_ptr<T, CopyPolicy, std::false_type>
		{
			auto p = static_cast<T*>(nullptr);
			auto pitch = std::size_t{};
			check(cudaMallocPitch(&p, &pitch, width * sizeof(T), height));
			return pitched_device_ptr<T, CopyPolicy, std::false_type>(detail::unique_device_ptr<T>(p), pitch, width, height);
		}

		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_host_ptr(std::size_t width, std::size_t height) -> pitched_host_ptr<T, CopyPolicy, std::false_type>
		{
			auto p = static_cast<T*>(nullptr);
			auto pitch = width * sizeof(T);
			check(cudaMallocHost(&p, pitch * height));
			return pitched_host_ptr<T, CopyPolicy, std::false_type>(detail::unique_host_ptr<T>(p), pitch, width, height);
		}

		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_device_ptr(std::size_t width, std::size_t height, std::size_t depth) -> pitched_device_ptr<T, CopyPolicy, std::true_type>
		{
			auto extent = make_cudaExtent(width * sizeof(T), height, depth);
			auto pitchedPtr = cudaPitchedPtr{};
			check(cudaMalloc3D(&pitchedPtr, extent));
			// omitting pitchedPtr.xsize and pitchedPtr.ysize as those are identical to width and height
			return pitched_device_ptr<T, CopyPolicy, std::true_type>(detail::unique_device_ptr<T>(pitchedPtr.ptr), pitchedPtr.pitch, width, height, depth);
		}

		template <class T, class CopyPolicy = sync_copy_policy>
		auto make_host_ptr(std::size_t width, std::size_t height, std::size_t depth) -> pitched_host_ptr<T, CopyPolicy, std::true_type>
		{
			auto p = static_cast<T*>(nullptr);
			auto pitch = width * sizeof(T);
			check(cudaMallocHost(&p, pitch * height * depth));
			return pitched_host_ptr<T, CopyPolicy, std::true_type>(detail::unique_host_ptr<T>(p), pitch, width, height, depth);
		}

		namespace detail
		{
			/* Functor for explicit copies */
			template <class CopyPolicy>
			struct copy_ftor : public CopyPolicy
			{
				/* 1D */
				template <class Dest, class Src>
				inline auto operator()(Dest& dest, const Src& src)
				-> typename std::enable_if<(!Dest::has_pitch && !Src::has_pitch), void>::type
				{
					CopyPolicy::copy(dest, src, src.size());
				}

				/* 2D */
				template <class Dest, class Src>
				inline auto operator()(Dest& dest, const Src& src)
				-> typename std::enable_if<(Dest::has_pitch && Src::has_pitch) && (!Dest::is3DPtr && !Src::is3DPtr), void>::type
				{
					CopyPolicy::copy(dest, src, src.width(), src.height());
				}

				/* 3D */
				template <class Dest, class Src>
				inline auto operator()(Dest& dest, const Src& src)
				-> typename std::enable_if<(Dest::has_pitch && Src::has_pitch) && (Dest::is3DPtr && Src::is3DPtr), void>::type
				{
					CopyPolicy::copy(dest, src, src.width(), src.height(), src.depth());
				}
			};
		}

		/*
		 * Explicit synchronous copy
		 */
		template <class Dest, class Src>
		auto copy_sync(Dest& dest, const Src& src) -> void
		{
			auto ftor = detail::copy_ftor<sync_copy_policy>{};
			ftor(dest, src);
		}

		/*
		 * Explicit asynchronous copy
		 */
		template <class Dest, class Src>
		auto copy_async(Dest& dest, const Src& src) -> void
		{
			auto ftor = detail::copy_ftor<async_copy_policy>{};
			ftor(dest, src);
		}
	}
}

#endif /* CUDA_MEMORY_H_ */
