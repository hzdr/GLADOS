#ifndef PIPELINE_PIPELINE_H_
#define PIPELINE_PIPELINE_H_

#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "Port.h"

namespace ddrf
{
	namespace pipeline
	{
		class Pipeline
		{
			public:
				template <class First, class Second>
				auto connect(First first, Second second) -> void
				{
					using port_type = typename First::element_type::output_type;
					auto port = std::unique_ptr<Port<port_type>>(new Port<port_type>);
					port->attach(second);
					first->attach(std::move(port));
				}

				template <class PipelineStage, typename... Args>
				auto create(Args&&... args) -> std::shared_ptr<PipelineStage>
				{
					return std::make_shared<PipelineStage>(std::forward<Args>(args)...);
				}

				template <class Stage>
				auto run(Stage stage) -> void
				{
					stage_threads_.emplace_back(&Stage::element_type::run, stage);
				}

				template <class Stage, class... Stages>
				auto run(Stage stage, Stages... stages) -> void
				{
					run(stage);
					run(stages...);
				}

				auto wait() -> void
				{
					for(auto&& t : stage_threads_)
						t.join();
				}

			private:
				std::vector<std::thread> stage_threads_;
		};

	}
}


#endif /* PIPELINE_PIPELINE_H_ */
