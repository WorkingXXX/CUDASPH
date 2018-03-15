
#include "smooth_kernel.cuh"

float* gHostM4Lut;
float* gDevM4Lut;

__host__ __device__ float M4Kernel(float kernel, float dist, float sr)
{
	float s = dist / sr;
	float result = kernel;

	if (s < 0.5)
	{
		result *= 1.0 - 6.0 * s * s + 6.0 * s * s * s;
	}
	else
	{
		result *= 2.0 * (1.0 - s) * (1.0 - s) * (1.0 - s);
	}

	return result;
}

__device__ float M4LutKernel(float kernel, float dist, float sr)
{
	int i = dist / sr * LUT_SIZE;

	if (i >= LUT_SIZE)
	{
		return 0.0;
	}
	else
	{
		return M4Kernel(kernel, sr * i / LUT_SIZE, sr);
	}
}

__device__ float Poly6Kernel(float kernel, float dist, float sr)
{
	float sr_2 = sr * sr;
	float dist_2 = dist * dist;
	return kernel * (sr_2 - dist_2) * (sr_2 - dist_2) * (sr_2 - dist_2);
}

__device__ float3 SpikyKernel1stDervt(float kernel, float3 distV, float sr)
{
	float dist = Distance(distV);
	return -distV * kernel * (sr - dist) * (sr - dist) / dist;
}

__device__ float LaplaceKernel2ndDervt(float kernel, float dist, float sr)
{
	return kernel * (sr - dist);
}
