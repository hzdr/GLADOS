#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <stdexcept>

namespace ddrf
{
	namespace cuda
	{
		class out_of_memory : public std::runtime_error
		{
			public:
				out_of_memory() : std::runtime_error{"out of memory"} {}
		};
	}
}



#endif /* EXCEPTION_H_ */
