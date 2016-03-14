#ifndef DDRF_IMAGELOADER_H_
#define DDRF_IMAGELOADER_H_

#include <memory>
#include <string>

#include "Image.h"

namespace ddrf
{
	template <class Implementation>
	class ImageLoader : public Implementation
	{
		public:
			using image_type = typename Implementation::image_type;

		public:
			/*
			 * Loads an image from the given path. The image data will be converted to the given
			 * data type if needed.
			 */
			template <typename T>
			auto loadImage(const std::string& path) -> decltype(Implementation::template loadImage<T>(path))
			{
				return Implementation::template loadImage<T>(path);
			}
	};
}



#endif /* DDRF_IMAGELOADER_H_ */
