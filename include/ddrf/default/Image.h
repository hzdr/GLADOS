#ifndef DEF_IMAGE_H_
#define DEF_IMAGE_H_

namespace ddrf
{
	namespace def
	{
		template <class T, class MemoryManager>
		class Image : public MemoryManager
		{
			public:
				using value_type = T;
				using pointer_type = typename MemoryManager::pointer_type;
				using size_type = typename MemoryManager::size_type;

			protected:
				~Image() = default;

				inline auto make_ptr(size_type width, size_type height) -> pointer_type
				{
					return MemoryManager::make_ptr(width, height);
				}

				template <class Source>
				inline auto copy(pointer_type& dest, Source& src, size_type width, size_type height) -> void
				{
					MemoryManager::copy(dest, src, width, height);
				}
		};
	}
}



#endif /* DEF_IMAGE_H_ */
