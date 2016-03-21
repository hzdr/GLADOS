#ifndef DDRF_IMAGESAVER_H_
#define DDRF_IMAGESAVER_H_

#include <string>
#include <utility>

#include "Image.h"
#include "Volume.h"

namespace ddrf
{
	template <class Implementation>
	class ImageSaver : public Implementation
	{
		public:
			using manager_type = typename Implementation::manager_type;

		public:
			/*
			 * Saves an image to the given path.
			 */
			auto saveImage(Image<manager_type> image, std::string path) -> void
			{
				Implementation::saveImage(std::move(image), path);
			}

			/*
			 * Saves a volume to the given path.
			 */
			auto saveVolume(Volume<manager_type> volume, std::string path) -> void
			{
				Implementation::saveVolume(std::move(volume), path);
			}

		protected:
			~ImageSaver() = default;

	};
}

#endif /* DDRF_IMAGESAVER_H_ */
