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
			using manager_type = typename Implementation::manager_type;

		public:
			/*
			 * Loads an image from the given path.
			 * */
			auto loadImage(const std::string& path) -> Image<manager_type>
			{
				return Implementation::loadImage(path);
			}
	};
}



#endif /* DDRF_IMAGELOADER_H_ */
