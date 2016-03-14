#ifndef PIPELINE_SOURCESTAGE_H_
#define PIPELINE_SOURCESTAGE_H_

#include <stdexcept>
#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK
#include <boost/log/trivial.hpp>

#include "../Filesystem.h"
#include "../Image.h"

#include "OutputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class ImageLoader>
		class SourceStage : public ImageLoader
						  , public OutputSide<Image<typename ImageLoader::image_type>>
		{
			public:
				using output_type = Image<typename ImageLoader::image_type>;

			public:
				SourceStage(const std::string& path)
				: ImageLoader(), OutputSide<output_type>(), path_{path}
				{
				}

				auto run() -> void
				{
					auto paths = readDirectory(path_);
					for(auto& path : paths)
					{
						auto img = ImageLoader::template loadImage<float>(path);
						if(img.valid())
							this->output(std::move(img));
						else
							BOOST_LOG_TRIVIAL(warning) << "SourceStage: Skipping invalid file " << path;
					}

					// all images loaded, send poisonous pill
					this->output(output_type());
				}

			private:
				std::string path_;
		};
	}
}


#endif /* PIPELINE_SOURCESTAGE_H_ */
