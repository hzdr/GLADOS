#ifndef DDRF_IMAGESAVER_H_
#define DDRF_IMAGESAVER_H_

namespace ddrf
{
	template <class Implementation>
	class ImageSaver : public Implementation
	{
		public:
			using image_type = typename Implementation::image_type;

		public:
			/*
			 * Saves an image to the given path.
			 */
			auto saveImage(Image<image_type>&& image, std::string path) -> void
			{
				Implementation::saveImage(std::forward<Image<image_type>&&>(image), path);
			}

			/*
			 * Saves an image into a volume at the given path.
			 */
			auto saveToVolume(Image<image_type>&& image, std::string path, std::size_t index) -> void
			{
				Implementation::saveToVolume(std::forward<Image<image_type>&&>(image),
												path, index);
			}

		protected:
			~ImageSaver() = default;

	};
}

#endif /* DDRF_IMAGESAVER_H_ */
