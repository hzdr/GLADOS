/*
 * HISHeader.h
 *
 *  Created on: 26.11.2015
 *      Author: Jan Stephan
 *
 *      Header definition for the HIS file format.
 *
 *      Types derived from old implementation:
 *			BYTE -> std::uint8_t
 *			DWORD -> std::uint32_t
 *      	WORD -> std::uint16_t
 *      	ULONG -> std::uint32_t
 */

#ifndef HISHEADER_H_
#define HISHEADER_H_

#include <cstdint>

namespace ddrf
{
	namespace loaders
	{
		enum class HISConst : std::uint16_t
		{
			file_header_size = 68,
			rest_size = 34,
			hardware_header_size = 32,
			header_size = file_header_size + hardware_header_size,
			file_id = 0x7000
		};

		struct HISHeader
		{
			std::uint16_t file_type;											// == HISConst::file_id
			std::uint16_t header_size;											// size of this file header in bytes
			std::uint16_t header_version;										// yy.y
			std::uint32_t file_size;											// size of the whole file in bytes
			std::uint16_t image_header_size;									// size of the image header in bytes
			std::uint16_t ulx, uly, brx, bry;									// bounding rectangle of image
			std::uint16_t number_of_frames;
			std::uint16_t correction;											// 0 = none, 1 = offset, 2 = gain, 4 = bad pixel, (ored)
			double integration_time;											// frame time in microseconds
			std::uint16_t type_of_numbers;										// short, long integer, float, signed/unsigned, inverted
																				// fault map, offset/gain correction data,
																				// badpixel correction data
			std::uint8_t x[static_cast<std::uint32_t>(HISConst::rest_size)];	// fill up to 68 byte
		};
	}
}


#endif /* HISHEADER_H_ */
