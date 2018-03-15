
#include "grid_cell_kernel.cuh"

__device__ bool IsGridIdxValid(int idx, int maxGridNum) 
{
	return !(idx == GRID_UNDEF || idx < 0 || idx > maxGridNum - 1);
}

__device__ int GetGridCell(
	const float3 & gridVolMin,
	const int3 & gridRes,
	const float3 & pos,
	float cellSize,
	int3 & gridCell)
{
	float gx = gridVolMin.x;
	float gy = gridVolMin.y;
	float gz = gridVolMin.z;

	int rx = gridRes.x;
	int ry = gridRes.y;
	int rz = gridRes.z;

	float px = pos.x - gx;
	float py = pos.y - gy;
	float pz = pos.z - gz;

	if (px < 0.0) px = 0.0;
	if (py < 0.0) py = 0.0;
	if (pz < 0.0) pz = 0.0;

	gridCell.x = (int)(px / cellSize);
	gridCell.y = (int)(py / cellSize);
	gridCell.z = (int)(pz / cellSize);

	if (gridCell.x > rx - 1) gridCell.x = rx - 1;
	if (gridCell.y > ry - 1) gridCell.y = ry - 1;
	if (gridCell.z > rz - 1) gridCell.z = rz - 1;

	return gridCell.y * rx * rz + gridCell.z * rx + gridCell.x;
}

__global__ void UniformAdd(
	int * data,
	int * uniforms,
	int n,
	int block_offset,
	int base_idx)
{
	__shared__ int uniform;

	if (threadIdx.x == 0)
		uniform = uniforms[blockIdx.x + block_offset];

	int address = threadIdx.x + __mul24(blockIdx.x, (blockDim.x << 1)) + base_idx;

	__syncthreads();

	data[address] += uniform;
	data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uniform;
}

__device__ int BuildSum(int * input)
{
	int idx = threadIdx.x;
	int stride = 1;

	// Build The Sum In Place Up The Tree
	for (int d = blockDim.x; d > 0; d >>= 1)
	{
		__syncthreads();

		if (idx < d)
		{
			int i = __mul24(__mul24(2, stride), idx);
			int ai = i + stride - 1;
			int bi = ai + stride;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			input[bi] += input[ai];
		}

		stride *= 2;
	}

	return stride;
}



__device__ void ScanRootToLeaves(int * input, int stride)
{
	int idx = threadIdx.x;

	// Traverse Down The Tree Building The Scan In Place
	for (int d = 1; d <= blockDim.x; d *= 2)
	{
		stride >>= 1;

		__syncthreads();

		if (idx < d)
		{
			int i = __mul24(__mul24(2, stride), idx);
			int ai = i + stride - 1;
			int bi = ai + stride;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int temp_data = input[ai];
			input[ai] = input[bi];
			input[bi] += temp_data;
		}

	}

}

