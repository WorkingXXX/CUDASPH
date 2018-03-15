
#ifndef GRID_CELL_KERNEL_CUH
#define GRID_CELL_KERNEL_CUH

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#define GRID_UCHAR   0xFF
#define GRID_UNDEF   0xFFFFFFFF
#define MAX_GRID_NBR 27

#define LOG_BANK_NUM 4
#define CONFLICT_FREE_OFFSET(index) ((index)>>LOG_BANK_NUM)

#define BLOCK_SIZE 256

// 判断单元格是否在范围内
__device__ bool IsGridIdxValid(int idx, int maxGridNum);

// 根据粒子的位置计算单元格的索引
__device__ int  GetGridCell(
	const float3 & gridVolMin,
	const int3 & gridRes,
	const float3 & pos,
	float cellSize,
	int3 & gridCell);

__global__ void UniformAdd(
	int * data, 
	int * uniforms, 
	int n, 
	int block_offset,
	int base_idx);


__device__ int BuildSum(int * input);


__device__ void ScanRootToLeaves(int * input, int stride);

template<bool is_np2>
__device__ void LoadSharedChunk(
	int * output, int * input,
	int n, int base_idx,
	int & ai, int & bi,
	int & mem_ai, int & mem_bi,
	int & bank_offset_a, int & bank_offset_b) 
{
	int i = threadIdx.x;

	mem_ai = base_idx + i;
	mem_bi = mem_ai + blockDim.x;

	ai = i;
	bi = i + blockDim.x;

	// Compute Spacing To Avoid Bank Conflicts
	bank_offset_a = CONFLICT_FREE_OFFSET(ai);
	bank_offset_b = CONFLICT_FREE_OFFSET(bi);

	// Cache The Computational Window In Shared Memory Pad Values Beyond N With Zeros
	output[ai + bank_offset_a] = input[mem_ai];

	if (is_np2) 
	{
		output[bi + bank_offset_b] = (bi < n) ? input[mem_bi] : 0;
	}
	else 
	{
		output[bi + bank_offset_b] = input[mem_bi];
	}

}

template<bool store_sum>
__device__ void ClearLastElement(int * input, int * block_sum, int block_idx)
{
	if (threadIdx.x == 0)
	{
		int i = (blockDim.x << 1) - 1;
		i += CONFLICT_FREE_OFFSET(i);

		if (store_sum)
		{
			// Write This Block's Total Sum To The Corresponding Index In The Block Sum Array
			block_sum[block_idx] = input[i];
		}
		// Zero The Last Element In The Scan So It Will propagate Back To The Front
		input[i] = 0;
	}

}

template<bool store_sum>
__device__ void PrescanBlock(int * data, int block_idx, int * block_sum)
{
	// Build The Sum In Place Up The Tree
	int stride = BuildSum(data);

	int b_idx = (block_idx == 0) ? blockIdx.x : block_idx;
	ClearLastElement<store_sum>(data, block_sum, b_idx);

	// Traverse Down Tree To Build The Scan
	ScanRootToLeaves(data, stride);
}

template<bool is_np2>
__device__ void StoreSharedChunk(
	int * output, int * input, 
	int n, int ai, int bi,
	int mem_ai, int mem_bi, 
	int bank_offset_a, int bank_offset_b)
{
	__syncthreads();

	// Write Results To Global Memoery
	output[mem_ai] = input[ai + bank_offset_a];

	if (is_np2)
	{
		if (bi < n)
		{
			output[mem_bi] = input[bi + bank_offset_b];
		}
	}
	else
	{
		output[mem_bi] = input[bi + bank_offset_b];
	}

}

template<bool store_sum, bool is_np2>
__global__ void PrescanCUDA(
	int * output, int * input,
	int * block_sum, 
	int n, int block_idx, int base_idx) 
{

	int ai, bi, mem_ai, mem_bi, bank_offset_a, bank_offset_b;

	extern __shared__ int shared_data[];

	int b_idx = (base_idx == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : base_idx;

	LoadSharedChunk<is_np2>(shared_data, input, n, b_idx, ai, bi, mem_ai, mem_bi, bank_offset_a, bank_offset_b);

	PrescanBlock<store_sum>(shared_data, block_idx, block_sum);

	StoreSharedChunk<is_np2>(output, shared_data, n, ai, bi, mem_ai, mem_bi, bank_offset_a, bank_offset_b);
}




#endif