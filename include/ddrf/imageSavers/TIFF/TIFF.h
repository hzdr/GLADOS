#ifndef SAVERS_TIFF_H_
#define SAVERS_TIFF_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <tiffio.h>

#include "../../Image.h"
#include "../../default/Image.h"

namespace ddrf
{
	namespace savers
	{
		template <typename T, class MemoryManager>
		class TIFF : public MemoryManager
		{
			public:
				using image_type = def::Image<T, MemoryManager>;

			public:
					auto saveImage(Image<image_type>&& image, std::string& path) const -> void
					{
						path.append(".tif");
						auto tif = TIFFOpen(path.c_str(), "w");
						if(tif == nullptr)
						{
							TIFFClose(tif);
							throw std::runtime_error("TIFFHandler: Could not open file " + path + " for writing.");
						}

						TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.width());
						TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.height());
						TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
						TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
						TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

						auto data = image.data();
						auto dataPtr = data;

						for(auto row = 0u; row < image.height(); ++row)
						{
							TIFFWriteScanline(tif, dataPtr, row);
							dataPtr += image.width();
						}

						TIFFClose(tif);
					}

					//TODO: saveToVolume

			protected:
					~TIFF() = default;
		};
	}
}

#endif /* SAVERS_TIFF_H_ */
