#ifndef SAVERS_TIFF_H_
#define SAVERS_TIFF_H_

#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <tiffio.h>

#include "../../Image.h"
#include "../../Volume.h"

namespace ddrf
{
	namespace savers
	{
		namespace detail
		{
			template<class T, bool = std::is_integral<T>::value, bool = std::is_unsigned<T>::value> struct SampleFormat {};
			template<class T> struct SampleFormat<T, true, true> { static constexpr auto value = SAMPLEFORMAT_UINT; };
			template<class T> struct SampleFormat<T, true, false> { static constexpr auto value = SAMPLEFORMAT_INT; };
			template<> struct SampleFormat<float> { static constexpr auto value = SAMPLEFORMAT_IEEEFP; };
			template<> struct SampleFormat<double>{ static constexpr auto value = SAMPLEFORMAT_IEEEFP; };

			template<class T> struct BitsPerSample { static constexpr auto value = sizeof(T) << 3; };

			struct TIFFDeleter { auto operator()(::TIFF* p) -> void { TIFFClose(p); }};
		}

		template <class MemoryManager>
		class TIFF : public MemoryManager
		{
			public:
				using manager_type = MemoryManager;

			public:
					auto saveImage(Image<MemoryManager> image, std::string& path) const -> void
					{
						path.append(".tif");
						// w8 enables Bigtiff
						auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter>{TIFFOpen(path.c_str(), "w8")};
						if(tif == nullptr)
							throw std::runtime_error{"savers::TIFF: Could not open file " + path + " for writing."};

						write_to_tiff(tif.get(), std::move(image));
					}

					auto saveVolume(Volume<MemoryManager> volume, std::string& path) const -> void
					{
						path.append(".tif");

						auto tif = std::unique_ptr<::TIFF, detail::TIFFDeleter>{TIFFOpen(path.c_str(), "w8")};
						if(tif == nullptr)
							throw std::runtime_error{"savers::TIFF: Could not open file " + path + " for writing."};

						for(auto i = 0u; i < volume.depth(); ++i)
						{
							auto slice = volume[i];

							write_to_tiff(tif.get(), std::move(slice));

							if(TIFFWriteDirectory(tif.get()) != 1)
								throw std::runtime_error{"savers::TIFF: tiffio error while writing to " + path};
						}
					}

			protected:
					~TIFF() = default;

			private:
					auto write_to_tiff(::TIFF* tif, Image<MemoryManager> img) const -> void
					{
						using value_type = typename Image<MemoryManager>::value_type;

						auto&& ss = std::stringstream{};
						// the locale will take ownership so plain new is okay here
						auto output_facet = new boost::posix_time::time_facet{"%Y:%m:%d %H:%M:%S"};

						ss.imbue(std::locale(std::locale::classic(), output_facet));
						ss.str("");

						auto now = boost::posix_time::second_clock::local_time();
						ss << now;

						TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.width());
						TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.height());
						TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, detail::BitsPerSample<value_type>::value);
						TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
						TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
						TIFFSetField(tif, TIFFTAG_THRESHHOLDING, THRESHHOLD_BILEVEL);
						TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
						TIFFSetField(tif, TIFFTAG_SOFTWARE, "ddrf");
						TIFFSetField(tif, TIFFTAG_DATETIME, ss.str().c_str());

						TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, detail::SampleFormat<value_type>::value);

						auto data = img.data();
						auto dataPtr = data;
						for(auto row = 0u; row < img.height(); ++row)
						{
							TIFFWriteScanline(tif, dataPtr, row);
							dataPtr += img.width();
						}
					}
		};
	}
}

#endif /* SAVERS_TIFF_H_ */
