#ifndef CUDA_COORDINATES_H_
#define CUDA_COORDINATES_H_

namespace ddrf
{
	namespace cuda
	{
		inline __device__ auto getX() -> unsigned int
		{
			return blockIdx.x * blockDim.x + threadIdx.x;
		}

		inline __device__ auto getY() -> unsigned int
		{
			return blockIdx.y * blockDim.y + threadIdx.y;
		}

		inline __device__ auto getZ() -> unsigned int
		{
			return blockIdx.z * blockDim.z + threadIdx.z;
		}
	}
}




#endif /* COORDINATES_H_ */
