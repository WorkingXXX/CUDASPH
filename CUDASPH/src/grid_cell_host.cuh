
#ifndef GRID_CELL_HOST_CUH
#define GRID_CELL_HOST_CUH

#include "math_support.cuh"
#include "grid_cell_kernel.cuh"

extern int ** blockSum;
extern int    elemAllocNum;
extern int    levelAllocNum;

// ¼ÆËãÇ°×ººÍ
void PrefixSumRecursive(
	int * output = 0,
	int * input = 0,
	int   elemNum = 0,
	int   level = 0);

void DeallocBlockSum();

void PreallocBlockSum(int maxElemNum);

bool IsPowerOf2(int n);

int  FloorPow2(int n);

#endif