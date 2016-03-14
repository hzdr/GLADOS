#ifndef DDRF_FILESYSTEM_H_
#define DDRF_FILESYSTEM_H_

#include <string>
#include <vector>

namespace ddrf
{
	auto readDirectory(const std::string&) -> std::vector<std::string>;
	auto createDirectory(const std::string&) -> bool;
}



#endif /* DDRF_FILESYSTEM_H_ */
