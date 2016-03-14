#ifndef PIPELINE_INPUTSIDE_H_
#define PIPELINE_INPUTSIDE_H_

#include <utility>

#include "../Queue.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class InputType>
		class InputSide
		{
			public:
				auto input(InputType&& in) -> void
				{
					input_queue_.push(std::forward<InputType&&>(in));
				}

			protected:
				Queue<InputType> input_queue_;
		};
	}
}


#endif /* PIPELINE_INPUTSIDE_H_ */
