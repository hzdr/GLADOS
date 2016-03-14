#ifndef PIPELINE_STAGE_H_
#define PIPELINE_STAGE_H_

#include <thread>
#include <utility>

#include "../Image.h"

#include "InputSide.h"
#include "OutputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class Implementation>
		class Stage
		: public InputSide<typename Implementation::input_type>
		, public OutputSide<typename Implementation::output_type>
		, public Implementation
		{
			public:
				using input_type = typename Implementation::input_type;
				using output_type = typename Implementation::output_type;

			public:
				template <typename... Args>
				Stage(Args&&... args)
				: InputSide<input_type>()
				, OutputSide<output_type>()
				, Implementation(std::forward<Args>(args)...)
				{
				}

				auto run() -> void
				{
					auto push_thread = std::thread{&Stage::push, this};
					auto take_thread = std::thread{&Stage::take, this};

					push_thread.join();
					take_thread.join();
				}

				auto push() -> void
				{
					while(true)
					{
						auto img = this->input_queue_.take();
						if(img.valid())
							Implementation::process(std::move(img));
						else
						{
							// received poisonous pill, time to die
							Implementation::process(std::move(img));
							break;
						}
					}
				}

				auto take() -> void
				{
					while(true)
					{
						auto result = Implementation::wait();
						if(result.valid())
							this->output(std::move(result));
						else
						{
							this->output(std::move(result));
							break;
						}
					}
				}
		};
	}
}


#endif /* PIPELINE_STAGE_H_ */
