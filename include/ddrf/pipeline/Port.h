#ifndef PIPELINE_PORT_H_
#define PIPELINE_PORT_H_

#include <memory>
#include <utility>

#include "InputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class DataType>
		class Port
		{
			public:
				void forward(DataType&& data)
				{
					next_->input(std::forward<DataType&&>(data));
				}

				void attach(std::shared_ptr<InputSide<DataType>> next) noexcept
				{
					next_ = next;
				}

			private:
				std::shared_ptr<InputSide<DataType>> next_;
		};
	}
}


#endif /* PIPELINE_PORT_H_ */
