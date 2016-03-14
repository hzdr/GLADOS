#ifndef PIPELINE_OUTPUTSIDE_H_
#define PIPELINE_OUTPUTSIDE_H_

#include <memory>
#include <utility>

#include "Port.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class OutputType>
		class OutputSide
		{
			public:
				auto output(OutputType&& out) -> void
				{
					if(port_ == nullptr)
						throw std::runtime_error("OutputSide: Missing port");

					port_->forward(std::forward<OutputType&&>(out));
				}

				auto attach(std::unique_ptr<Port<OutputType>>&& port) noexcept -> void
				{
					port_ = std::move(port);
				}

			protected:
				std::unique_ptr<Port<OutputType>> port_;
		};
	}
}


#endif /* PIPELINE_OUTPUTSIDE_H_ */
