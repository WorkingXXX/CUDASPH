
#include "grid_cell_host.cuh"

int ** blockSum;
int    elemAllocNum;
int    levelAllocNum;

void PrefixSumRecursive(
	int * output,
	int * input,
	int   elemNum,
	int   level)
{
	const int blockSize = BLOCK_SIZE;
	const int bankNum = 32;

	int blockNum = Max(1, (int)ceil((float)elemNum / (2.0f * blockSize)));
	int threadNum;

	if (blockNum > 1)
	{
		threadNum = blockSize;
	}
	else if (IsPowerOf2(elemNum))
	{
		threadNum = elemNum / 2;
	}
	else
	{
		threadNum = FloorPow2(elemNum);
	}

	int elemNumPerBlock = threadNum * 2;
	int elemNumLastBlock = elemNum - (blockNum - 1) * elemNumPerBlock;
	int threadNumLastBlock = Max(1, elemNumLastBlock / 2);
	int np2LastBlock = 0;
	int sharedMemLastBlock = 0;

	if (elemNumLastBlock != elemNumPerBlock)
	{
		np2LastBlock = 1;

		if (!IsPowerOf2(elemNumLastBlock))
		{
			threadNumLastBlock = FloorPow2(elemNumLastBlock);
		}

		int extraSpace = (2 * threadNumLastBlock) / bankNum;
		sharedMemLastBlock = SIZE_FLT * (2 * threadNumLastBlock + extraSpace);
	}

	int extraSpace = elemNumPerBlock / bankNum;
	int sharedMemSize = SIZE_FLT * (elemNumPerBlock + extraSpace);

	dim3 grids(Max(1, blockNum - np2LastBlock), 1, 1);
	dim3 threads(threadNum, 1, 1);

	/*printf("Config In PrefixSumRecursive:\n");
	printf("grids   : (%d, %d, %d)\n", grids.x, grids.y, grids.z);
	printf("threads : (%d, %d, %d)\n", threads.x, threads.y, threads.z);
	printf("shared mem : %d\n", shared_mem_size);*/

	if (blockNum > 1)
	{
		PrescanCUDA<true, false> << <grids, threads, sharedMemSize >> > (output, input, blockSum[level], threadNum * 2, 0, 0);

		if (np2LastBlock)
		{
			PrescanCUDA<true, true> << <1, threadNumLastBlock, sharedMemLastBlock >> >(output, input, blockSum[level],
				elemNumLastBlock, blockNum - 1, elemNum - elemNumLastBlock);
		}

		PrefixSumRecursive(blockSum[level], blockSum[level], blockNum, level + 1);

		UniformAdd << <grids, threads >> > (output, blockSum[level],
			elemNum - elemNumLastBlock, 0, 0);

		if (np2LastBlock)
		{
			UniformAdd << <1, threadNumLastBlock >> >(output, blockSum[level], elemNumLastBlock,
				blockNum - 1, elemNum - elemNumLastBlock);
		}

	}
	else if (IsPowerOf2(elemNum))
	{
		PrescanCUDA<false, false> << <grids, threads, sharedMemSize >> > (output, input, 0, threadNum * 2, 0, 0);
	}
	else
	{
		PrescanCUDA<false, true> << <grids, threads, sharedMemSize >> > (output, input, 0, elemNum, 0, 0);
	}
}

void DeallocBlockSum()
{
	for (int i = 0; i < levelAllocNum; i++)
	{
		cudaFree(blockSum[i]);
	}
	free((void**)blockSum);

	blockSum = 0;
	elemAllocNum = 0;
	levelAllocNum = 0;
}

void PreallocBlockSum(int max_elem_num)
{
	elemAllocNum = max_elem_num;

	int block_size = BLOCK_SIZE;
	int elem_num = max_elem_num;
	int level = 0;

	do
	{
		int block_num = Max(1, (int)ceil((float)elem_num / (2.0f * block_size)));

		if (block_num > 1) level++;

		elem_num = block_num;

	} while (elem_num > 1);

	blockSum = (int**)malloc(level * sizeof(int*));
	levelAllocNum = level;

	elem_num = max_elem_num;
	level = 0;

	do {

		int block_num = Max(1, (int)ceil((float)elem_num / (2.0f * block_size)));

		if (block_num > 1)
		{
			cudaMalloc((void**)&blockSum[level++], block_num * SIZE_INT);
		}

		elem_num = block_num;

	} while (elem_num > 1);
}

inline bool IsPowerOf2(int n)
{
	return ((n & (n - 1)) == 0);
}

inline int FloorPow2(int n)
{
	int exp;
	frexp((float)n, &exp);
	return 1 << (exp - 1);
}