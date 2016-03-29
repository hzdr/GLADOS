#ifndef PIPELINE_SINKSTAGE_H_
#define PIPELINE_SINKSTAGE_H_

#include <cstdint>
#include <string>
#include <utility>

#include <boost/log/trivial.hpp>

#include "../Filesystem.h"
#include "../Image.h"

#include "InputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class ImageSaver>
		class SinkStage : public ImageSaver
						, public InputSide<ddrf::Volume<typename ImageSaver::manager_type>>
		{
			public:
				using input_type = ddrf::Volume<typename ImageSaver::manager_type>;

			public:
				SinkStage(const std::string& path, const std::string& prefix)
				: ImageSaver(), InputSide<input_type>(), path_{path}, prefix_{prefix}
				{
					bool created = createDirectory(path_);
					if(!created)
						BOOST_LOG_TRIVIAL(fatal) << "SinkStage: Could not create target directory at " << path;

					if(path_.back() != '/')
						path_.append("/");
				}

				auto run() -> void
				{
					auto counter = 0;
					while(true)
					{
						auto vol = this->input_queue_.take();
						if(vol.valid())
						{
							auto path = path_ + prefix_ + std::to_string(counter);
							BOOST_LOG_TRIVIAL(debug) << "SinkStage: Saving to " << path;
							ImageSaver::saveVolume(std::move(vol), path);
							++counter;
						}
						else
						{
							BOOST_LOG_TRIVIAL(debug) << "SinkStage: Poisonous pill arrived, terminating.";
							break; // poisonous pill
						}
					}
				}

			private:
				std::string path_;
				std::string prefix_;
		};
	}
}


#endif /* PIPELINE_SINKSTAGE_H_ */
