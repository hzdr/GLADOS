#ifndef LAUNCH_H_
#define LAUNCH_H_

#include <cstddef>
#include <cstdint>

#include "Check.h"

namespace ddrf
{
	namespace cuda
	{
		namespace detail
		{
			inline auto roundUp(std::uint32_t num, std::uint32_t multiple) -> std::uint32_t
			{
				if(multiple == 0)
					return num;

				if(num == 0)
					return multiple;

				auto remainder = num % multiple;
				if(remainder == 0)
					return num;

				return num + multiple - remainder;
			}
		}

		template <typename... Args>
		auto launch(std::size_t input_size, void(*kernel)(Args...), Args... args) -> void
		{
			// calculate max potential blocks
			auto block_size = int{};
			auto min_grid_size = int{};
			CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0));

			// calculate de facto occupation based on input size
			auto grid_size = (input_size + block_size - 1) / block_size;

			kernel<<<grid_size, block_size>>>(std::forward<Args>(args)...);
			CHECK(cudaPeekAtLastError());
		}

		template <typename... Args>
		auto launch(std::size_t input_width, std::size_t input_height, void(*kernel)(Args...), Args... args) -> void
		{
			auto threads = detail::roundUp(static_cast<unsigned int>(input_width * input_height), 1024u);
			auto blocks = threads / 1024u;

			auto iwb = static_cast<unsigned int>(input_width) / blocks;
			auto dim_x = ((iwb < 32u) && (iwb != 0u)) ? iwb : detail::roundUp(iwb, 32u);
			auto ihb = static_cast<unsigned int>(input_height) / blocks;
			auto dim_y = ((ihb < 32u) && (ihb != 0u)) ? ihb : detail::roundUp(ihb, 32u);
			auto block_size = dim3{dim_x, dim_y};
			auto grid_size = dim3{static_cast<unsigned int>((input_width + block_size.x - 1u)/block_size.x),
									static_cast<unsigned int>((input_height + block_size.y - 1u)/block_size.y)};

			kernel<<<grid_size, block_size>>>(args...);
			CHECK(cudaPeekAtLastError());
		}

		template <typename... Args>
		auto launch(std::size_t input_width, std::size_t input_height, std::size_t input_depth, void(*kernel)(Args...), Args... args) -> void
		{
			auto threads = detail::roundUp(static_cast<unsigned int>(input_width * input_height * input_depth), 1024u);
			auto blocks = threads / 1024u;

			auto iwb = static_cast<unsigned int>(input_width) / blocks;
			auto dim_x = ((iwb < 16u) && (iwb != 0u)) ? iwb : detail::roundUp(iwb, 16u);
			auto ihb = static_cast<unsigned int>(input_height) / blocks;
			auto dim_y = ((ihb < 16u) && (ihb != 0u)) ? ihb : detail::roundUp(ihb, 16u);
			auto idb = static_cast<unsigned int>(input_depth) / blocks;
			auto dim_z = ((idb < 4u) && (idb != 0u)) ? idb : detail::roundUp(idb, 4u);
			auto block_size = dim3{dim_x, dim_y, dim_z};
			auto grid_size = dim3{static_cast<unsigned int>((input_width + block_size.x - 1) / block_size.x),
									static_cast<unsigned int>((input_height + block_size.y - 1) / block_size.y),
									static_cast<unsigned int>((input_depth + block_size.z - 1) / block_size.z)};

			kernel<<<grid_size, block_size>>>(args...);
			CHECK(cudaPeekAtLastError());
		}
	}
}



#endif /* LAUNCH_H_ */
