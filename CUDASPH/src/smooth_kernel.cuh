
#ifndef SMOOTH_KERNEL_CUH
#define SMOOTH_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include "math_support.cuh"

#define LUT_SIZE  100000
#define LUT_CONST 2.546479089470325472f

extern float* gHostM4Lut;
extern float* gDevM4Lut;

__host__ __device__ float M4Kernel(float kernel, float dist, float sr);

__device__ float M4LutKernel(float kernel, float dist, float sr);

__device__ float Poly6Kernel(float kernel, float dist, float sr);

__device__ float3 SpikyKernel1stDervt(float kernel, float3 distV, float sr);

__device__ float LaplaceKernel2ndDervt(float kernel, float dist, float sr);

#endif