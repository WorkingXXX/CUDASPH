
#include "fluid_system_cuda.cuh"

intptr     gDevPtcGridIdx;    // 粒子所在单元格的索引
intptr     gDevGridPtcIdx;    // 单元格中粒子的索引
intptr     gDevGridPtcNum;    // 单元格所含有的粒子数目
intptr     gDevGridPtcOffset; // 单元格中粒子的偏移
intptr     gDevGridOffset;    // 单元格偏移

flt3ptr gHostPos;
flt3ptr gHostVel;
PtcAttrPtr gHostPtcAttr;

FluidConst gHostFldCst;

void AdjustBlockAndThreadNum(int& blockNum, int& threadNum)
{
	threadNum = 128;
	blockNum = (gHostFldCst.ptcTotalNum / threadNum) + 1;
}

void PrintDeviceProperty()
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);	
	printf("--------------------------------------------------------------------\n");
	printf("Device Property Can Be Listed As Followed:\n");
	printf("Device Name              : %s\n", devProp.name);
	printf("Compute Capability       : %d.%d\n", devProp.major, devProp.minor);
	printf("Total Global Mem         : %u\n", devProp.totalGlobalMem);
	printf("Total Const Mem          : %u\n", devProp.totalConstMem);
	printf("Max Grid Of Each Dim     : (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	printf("Max Block Of Each Dim    : (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
	printf("Max Thread Per Block     : %u\n", devProp.maxThreadsPerBlock);
	printf("Max Shared Mem Per Block : %u\n", devProp.sharedMemPerBlock);
	printf("--------------------------------------------------------------------\n");
}

void AllocGlobalData(int ptcTotalNum, int gridTotalNum)
{
	cudaMalloc(&gDevPos, ptcTotalNum * SIZE_FLT3);
	cudaMalloc(&gDevVel, ptcTotalNum * SIZE_FLT3);
	cudaMalloc(&gDevAcc, ptcTotalNum * SIZE_FLT3);
	cudaMalloc(&gDevPress, ptcTotalNum * SIZE_FLT);
	cudaMalloc(&gDevPtcAttr, ptcTotalNum * sizeof(PtcAttrib));
	cudaMalloc(&gDevPtcGridIdx, ptcTotalNum * SIZE_INT);
	cudaMalloc(&gDevGridPtcIdx, ptcTotalNum * SIZE_INT);
	cudaMalloc(&gDevGridPtcOffset, ptcTotalNum * SIZE_INT);
	cudaMalloc(&gDevGridPtcNum, gridTotalNum * SIZE_INT);
	cudaMalloc(&gDevGridOffset, gridTotalNum * SIZE_INT);

	cudaMalloc(&gDevPBFInterPos, ptcTotalNum * SIZE_FLT3);
	cudaMalloc(&gDevPBFFactor, ptcTotalNum * SIZE_FLT);

	PrintCUDAError("AllocGlobalData");
}

void InitGlobalData(int ptcTotalNum, int gridTotalNum)
{
	cudaMemcpy(gDevPos, gHostPos, ptcTotalNum * SIZE_FLT3, cudaMemcpyHostToDevice);

	cudaMemcpy(gDevVel, gHostVel, ptcTotalNum * SIZE_FLT3, cudaMemcpyHostToDevice);

	cudaMemset(gDevAcc, 0, ptcTotalNum * SIZE_FLT3);

	cudaMemset(gDevPress, 0, ptcTotalNum * SIZE_FLT);

	cudaMemcpy(gDevPtcAttr, gHostPtcAttr, ptcTotalNum * sizeof(PtcAttrib), cudaMemcpyHostToDevice);

	cudaMemset(gDevPtcGridIdx, GRID_UNDEF, ptcTotalNum * SIZE_INT);

	cudaMemset(gDevGridPtcIdx, GRID_UCHAR, ptcTotalNum * SIZE_INT);

	cudaMemset(gDevGridPtcOffset, 0, ptcTotalNum * SIZE_INT);

	cudaMemset(gDevGridPtcNum, 0, gridTotalNum * SIZE_INT);

	cudaMemset(gDevGridOffset, 0, gridTotalNum * SIZE_INT);

	cudaMemcpyToSymbol(gDevFldCst, &gHostFldCst, sizeof(FluidConst));

	cudaDeviceSynchronize();

	DeallocBlockSum();

	PreallocBlockSum(gridTotalNum);

	PrintCUDAError("InitGlobalData");
}

void FreeGlobalData()
{
	cudaFreeHost(gHostPos);
	cudaFreeHost(gHostVel);
	cudaFreeHost(gHostPtcAttr);
	cudaFree(gDevPos);
	cudaFree(gDevVel);
	cudaFree(gDevAcc);
	cudaFree(gDevPress);
	cudaFree(gDevPtcAttr);
	cudaFree(gDevPtcGridIdx);
	cudaFree(gDevGridPtcIdx);
	cudaFree(gDevGridPtcOffset);
	cudaFree(gDevGridPtcNum);
	cudaFree(gDevGridOffset);
}

//#define PRINT_DEBUG_INFO

void InitSimulation(Scene scene)
{
	PrintDeviceProperty();

	float3 gridMin = make_float3(0.0, 0.0, 0.0);
	float3 gridMax = make_float3(50.0, 50.0, 50.0);
	float3 fluidMin = gridMin;
	float3 fluidMax = gridMax * 0.6;

	InitFluidConst(gHostFldCst);

	PreComputeSpacing(gHostFldCst);

	PreComputeKernels(gHostFldCst);

	PreComputeGasConstantAndTimeStep(gHostFldCst);

	CreateGrid(scene, gridMin, gridMax, gHostFldCst.cellSize / gHostFldCst.simuScale);

	CreateFluid(scene, fluidMin, fluidMax, gHostFldCst.spaceGrpc);

	InitParticleAttrib(scene, gHostFldCst.ptcTotalNum);

	AllocGlobalData(gHostFldCst.ptcTotalNum, gHostFldCst.gridTotalNum);

	InitGlobalData(gHostFldCst.ptcTotalNum, gHostFldCst.gridTotalNum);

	AdjustBlockAndThreadNum(gBlockNum, gThreadNum);

	printf("--------------------------------------------------------------------\n");
	printf("Simulation Config Can Be Listed As Followed : \n");
	printf("ptcTotalNum : %d\n", gHostFldCst.ptcTotalNum);
	printf("gridTotalNum: %d\n", gHostFldCst.gridTotalNum);
	printf("cellSize    : %.6f\n", gHostFldCst.cellSize);
	printf("spaceGrpc   : %.6f\n", gHostFldCst.spaceGrpc);
	printf("sr          : %.6f\n", gHostFldCst.sr);
	printf("timeStep    : %.6f\n", gHostFldCst.timeStep);
	printf("poly6Kernel : %.6f\n", gHostFldCst.poly6Kernel);
	printf("spikyKernel : %.6f\n", gHostFldCst.spikyKernel);
	printf("lapKernel   : %.6f\n", gHostFldCst.laplaceKernel);
	printf("lutKernel   : %.6f\n", gHostFldCst.lutKernel);
	printf("blockNum    : %d\n", gBlockNum);
	printf("threadNum   : %d\n", gThreadNum);
	printf("--------------------------------------------------------------------\n");
}

void InitParticleAttrib(Scene scene, int ptcTotalNum)
{
	cudaHostAlloc(&gHostPtcAttr, ptcTotalNum * sizeof(PtcAttrib), cudaHostAllocDefault);

	for (size_t i = 0; i < ptcTotalNum; ++i)
	{
		gHostPtcAttr[i] = CreateNormalParticle();
	}
}

void CreateGrid(
	Scene scene,
	const float3& gridMin,
	const float3& gridMax,
	float cellSize)
{
	gHostFldCst.gridMin = gridMin;
	gHostFldCst.gridMax = gridMax;
	gHostFldCst.gridSize = gridMax - gridMin;
	gHostFldCst.gridRes = make_int3(
		(int)ceil(gHostFldCst.gridSize.x / cellSize),
		(int)ceil(gHostFldCst.gridSize.y / cellSize),
		(int)ceil(gHostFldCst.gridSize.z / cellSize));

	if (gHostFldCst.gridRes.x == 0) gHostFldCst.gridRes.x = 1;
	if (gHostFldCst.gridRes.y == 0) gHostFldCst.gridRes.y = 1;
	if (gHostFldCst.gridRes.z == 0) gHostFldCst.gridRes.z = 1;

	gHostFldCst.gridTotalNum = gHostFldCst.gridRes.x * gHostFldCst.gridRes.y * gHostFldCst.gridRes.z;

	int cellIdx = 0;
	for (int y = -1; y < 2; ++y)
	{
		for (int x = -1; x < 2; ++x)
		{
			for (int z = -1; z < 2; ++z)
			{
				gHostFldCst.nbrOffset[cellIdx++] =
					y * gHostFldCst.gridRes.x * gHostFldCst.gridRes.z +
					z * gHostFldCst.gridRes.x + x;
			}
		}
	}

}

void CreateFluid(
	Scene scene,
	const float3& fluidMin,
	const float3 & fluidMax,
	float spacing)
{
	gHostFldCst.fluidInitMin = fluidMin;
	gHostFldCst.fluidInitMax = fluidMax;
	gHostFldCst.fluidInitSize = fluidMax - fluidMin;
	gHostFldCst.fluidInitRes = make_int3(
		(int)ceil(gHostFldCst.fluidInitSize.x / spacing),
		(int)ceil(gHostFldCst.fluidInitSize.y / spacing),
		(int)ceil(gHostFldCst.fluidInitSize.z / spacing));
	gHostFldCst.ptcTotalNum = gHostFldCst.fluidInitRes.x * gHostFldCst.fluidInitRes.y * gHostFldCst.fluidInitRes.z;

	srand(time(0));

	float spaceOffsetX = 0.0, spaceOffsetY = 0.0, spaceOffsetZ = 0.0;
	if (gHostFldCst.fluidInitRes.x % 2) spaceOffsetX = 0.5;
	if (gHostFldCst.fluidInitRes.y % 2) spaceOffsetY = 0.5;
	if (gHostFldCst.fluidInitRes.z % 2) spaceOffsetZ = 0.5;

	cudaHostAlloc(&gHostPos, gHostFldCst.ptcTotalNum * SIZE_FLT3, cudaHostAllocDefault);
	cudaHostAlloc(&gHostVel, gHostFldCst.ptcTotalNum * SIZE_FLT3, cudaHostAllocDefault);

	float jitter = 0.1;

	int i = 0;
	for (size_t y = 0; y < gHostFldCst.fluidInitRes.y; ++y)
	{
		float py = fluidMin.y +spacing * 0.5 + (y + spaceOffsetY) * spacing;
		for (size_t x = 0; x < gHostFldCst.fluidInitRes.x; ++x)
		{
			float px = fluidMin.x + spacing * 0.5 + (x + spaceOffsetX) * spacing;
			for (size_t z = 0; z < gHostFldCst.fluidInitRes.z; ++z)
			{
				float pz = fluidMin.z + spacing * 0.5 + (z + spaceOffsetZ) * spacing;

				if (i < gHostFldCst.ptcTotalNum)
				{
					gHostPos[i] = make_float3(px + (NormalRand() - 0.5) * jitter, py + (NormalRand() - 0.5) * jitter, pz + (NormalRand() - 0.5) * jitter);
					gHostVel[i++] = make_float3(0.0, 0.0, 0.0);
				}
			}
		}
	}

}

void PreComputeKernels(FluidConst & fldCst)
{
	float sr = fldCst.sr;

	fldCst.poly6Kernel = 315.0 / (64.0 * MY_PI * pow(sr, 9));
	fldCst.spikyKernel = -45.0 / (MY_PI * pow(sr, 6));
	fldCst.laplaceKernel = -fldCst.spikyKernel;
	fldCst.lutKernel = LUT_CONST / pow(sr, 3);

	fldCst.densContrib = M4Kernel(fldCst.lutKernel, 0.0, sr);

	/*for (size_t i = 0; i < LUT_SIZE; ++i)
	{
	float dist = sr * i / LUT_SIZE;
	gHostM4Lut[i] = M4Kernel(fldCst.lutKernel, dist, sr);
	}*/
}

void PreComputeSpacing(FluidConst & fldCst)
{
	//fldCst.simuScale = 1.0;
	//fldCst.spaceReal = 1.0;
	//fldCst.ptcMass = fldCst.restDens * pow(fldCst.spaceReal, 3.0);
	//// 平滑核半径为粒子间初始距离的两倍，为了减少误差而设置成2.002倍
	//fldCst.sr = fldCst.spaceReal * 2.0;
	//fldCst.cellSize = fldCst.sr;
	//fldCst.ptcRadius = fldCst.spaceReal * 0.5;
}

void PreComputeGasConstantAndTimeStep(FluidConst & fldCst)
{
	// 根据CFL条件计算出最大时间步长
	fldCst.k = 2000.0;
	float maxSpeed = sqrt(2 * GRAVITY * (fldCst.gridMax.y - fldCst.gridMin.y));
	float soundSpeed = sqrt(fldCst.k);
	fldCst.timeStep = CFL_FACTOR * fldCst.sr / Max(maxSpeed, soundSpeed);
	fldCst.timeStep = 0.016;
}
//#define PRINT_DEBUG_INFO
void FetchDataFromGPU()
{
	cudaMemcpy(gHostPos, gDevPos, gHostFldCst.ptcTotalNum * SIZE_FLT3, cudaMemcpyDeviceToHost);

	PrintCUDAError("FetchDataFromGPU");
}

void InsertParticles()
{
	cudaMemset(gDevGridPtcNum, 0, gHostFldCst.gridTotalNum * SIZE_INT);

	InsertParticleCUDA << <gBlockNum, gThreadNum >> >(
		gDevPos,
		gDevPtcGridIdx,
		gDevGridPtcNum,
		gDevGridPtcOffset);

	cudaDeviceSynchronize();

	PrintCUDAError("InsertParticles");
}

void FindNbrParticles()
{
	InsertParticles();
	PrefixSum();
	CountingSort();
}

void PrefixSum()
{
	PrefixSumRecursive(
		gDevGridOffset,
		gDevGridPtcNum,
		gHostFldCst.gridTotalNum,
		0);

	cudaDeviceSynchronize();

	PrintCUDAError("PrefixSum");
}

void CountingSort()
{
	cudaMemset(gDevGridPtcIdx, GRID_UCHAR, gHostFldCst.ptcTotalNum * SIZE_INT);
	
	CountingSortCUDA << <gBlockNum, gThreadNum >> >(
		gDevPtcGridIdx,
		gDevGridPtcOffset,
		gDevGridOffset,
		gDevGridPtcIdx);
	
	cudaDeviceSynchronize();

	PrintCUDAError("CountingSort");
}


void ComputeDensAndPress()
{
	ComputeDensAndPressCUDA << <gBlockNum, gThreadNum >> >(
		gDevPos,
		gDevPress,
		gDevPtcAttr,
		gDevPtcGridIdx,
		gDevGridPtcNum,
		gDevGridOffset,
		gDevGridPtcIdx);

	cudaDeviceSynchronize();

	PrintCUDAError("ComputeDensAndPress");
}

void ComputeAcceleration()
{
	ComputeAccelerationCUDA << <gBlockNum, gThreadNum >> >(
		gDevPos,
		gDevVel,
		gDevAcc,
		gDevPress,
		gDevPtcAttr,
		gDevPtcGridIdx,
		gDevGridPtcNum,
		gDevGridOffset,
		gDevGridPtcIdx);

	cudaDeviceSynchronize();

	PrintCUDAError("ComputeAcceleration");
}

void AdvanceParticles()
{
	AdvanceParticleCUDA << <gBlockNum, gThreadNum >> >(
		gDevPos,
		gDevVel,
		gDevAcc);

	cudaDeviceSynchronize();

	PrintCUDAError("AdvanceParticles");
}

void StandardSPH()
{
	FindNbrParticles();
	ComputeDensAndPress();
	ComputeAcceleration();
	AdvanceParticles();
	FetchDataFromGPU();
}

void PBF_PredictPosition()
{
	//cudaMemset(gDevPBFInterPos, 0, gHostFldCst.ptcTotalNum);

	PBF_PredictPositionCUDA << <gBlockNum, gThreadNum >> > (
		gDevPos,
		gDevPBFInterPos,
		gDevVel,
		gDevAcc);

	cudaDeviceSynchronize();

	PrintCUDAError("PBF_PredictPosition");
}

void PBF_ComputeDensAndFactor()
{
	cudaMemset(gDevPBFFactor, 0, gHostFldCst.ptcTotalNum);
	
	PBF_ComputeDensAndFactorCUDA << <gBlockNum, gThreadNum >> > (
		gDevPBFInterPos,
		gDevPBFFactor,
		gDevPtcAttr,
		gDevPtcGridIdx,
		gDevGridPtcNum,
		gDevGridOffset,
		gDevGridPtcIdx);

	cudaDeviceSynchronize();

	static int i = 1;
	if (i-- > 0)
	{
		flt3ptr testInterPos = new float3[gHostFldCst.ptcTotalNum];
		fltptr  testFactor = new float[gHostFldCst.ptcTotalNum];
		PtcAttrPtr testAttr = new PtcAttrib[gHostFldCst.ptcTotalNum];
		cudaMemcpy(testInterPos, gDevPBFInterPos, gHostFldCst.ptcTotalNum * SIZE_FLT3, cudaMemcpyDeviceToHost);
		cudaMemcpy(testFactor, gDevPBFFactor, gHostFldCst.ptcTotalNum * SIZE_FLT, cudaMemcpyDeviceToHost);
		cudaMemcpy(testAttr, gDevPtcAttr, gHostFldCst.ptcTotalNum * sizeof(PtcAttrib), cudaMemcpyDeviceToHost);
		for (int i = 0; i < gHostFldCst.ptcTotalNum; i += 100)
		{
			printf("%d : nbrCnt=%d, factor=%.10f, dens=%.10f\n", i, testAttr[i].nbrCnt, testFactor[i], testAttr[i].dens);
		}
		delete[] testInterPos;
		delete[] testFactor;
		delete[] testAttr;
	}

	PrintCUDAError("PBF_ComputeDensAndFactor");
}

void PBF_ComputePositionDelta()
{
	PBF_ComputePositionDeltaCUDA << <gBlockNum, gThreadNum >> > (
		gDevPBFInterPos,
		gDevVel,
		gDevPBFFactor,
		gDevPtcAttr,
		gDevPtcGridIdx,
		gDevGridPtcNum,
		gDevGridOffset,
		gDevGridPtcIdx);

	cudaDeviceSynchronize();

	static int i = 1;
	if (i-- > 0)
	{
		flt3ptr testInterPos = new float3[gHostFldCst.ptcTotalNum];
		cudaMemcpy(testInterPos, gDevPBFInterPos, gHostFldCst.ptcTotalNum * SIZE_FLT3, cudaMemcpyDeviceToHost);
		for (int i = 0; i < gHostFldCst.ptcTotalNum; i += 100)
		{
			printf("%d : interPos=(%.16f,%.16f,%.16f\n)", i, testInterPos[i].x, testInterPos[i].y, testInterPos[i].z);
		}
		delete[] testInterPos;
	}

	PrintCUDAError("PBF_ComputePositionDelta");
}

void PBF_UpdateVelocity()
{
	PBF_UpdateVelocityCUDA << <gBlockNum, gThreadNum >> > (
		gDevPos,
		gDevPBFInterPos,
		gDevVel);

	cudaDeviceSynchronize();

	PrintCUDAError("PBF_UpdateVelocity");
}

void PBF_XSPHAndVorticity()
{
	PBF_XSPHAndVorticityCUDA << <gBlockNum, gThreadNum >> > ();

	cudaDeviceSynchronize();
	
	PrintCUDAError("PBF_XSPHAndVorticity");
}


void PBF_BoundaryConstraint()
{
	PBF_BoundaryConstraintCUDA << <gBlockNum, gThreadNum >> > (
		gDevPBFInterPos,
		gDevVel);

	cudaDeviceSynchronize();

	PrintCUDAError("PBF_XSPHAndVorticity");
}

#define PBF_ITERATION 4

void PBF()
{
	PBF_BoundaryConstraint();
	PBF_PredictPosition();

	FindNbrParticles();

	int iter = PBF_ITERATION;
	while (iter--)
	{
		PBF_ComputeDensAndFactor();
		PBF_ComputePositionDelta();	
	}

	PBF_BoundaryConstraint();
	PBF_UpdateVelocity();
	PBF_XSPHAndVorticity();

	FetchDataFromGPU();
}

static int stepCnt = 0;
void SimulateOnGPU()
{
#ifdef PRINT_DEBUG_INFO
	printf("Step %d\n", stepCnt++);
#endif
	PBF();
	//StandardSPH();
}

void PrintCUDAError(const char* funcName)
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA Error In Function %s : %s - %s\n", funcName,
			cudaGetErrorName(error), cudaGetErrorString(error));
	}
}

__global__ void CountingSortCUDA(
	intptr ptcGridIdx,
	intptr gridPtcOffset,
	intptr gridOffset,
	intptr gridPtcIdx)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int pgIdx = ptcGridIdx[i];

	if (pgIdx != GRID_UNDEF)
	{
		int ptcOffset = gridPtcOffset[i];
		int sortIdx = gridOffset[pgIdx] + ptcOffset;

		//if (sortIdx < gDevFldCst[0].ptcTotalNum)
		gridPtcIdx[sortIdx] = i;
	}
}

__global__ void InsertParticleCUDA(
	flt3ptr pos,
	intptr  ptcGridIdx,
	intptr  gridPtcNum,
	intptr  gridPtcOffset)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int3 gridCell = make_int3(0, 0, 0);

	int gcIdx = GetGridCell(
		gDevFldCst[0].gridMin,
		gDevFldCst[0].gridRes,
		pos[i],
		gDevFldCst[0].cellSize / gDevFldCst[0].simuScale,
		gridCell);

	if (make_int3(0, 0, 0) <= gridCell && gridCell < gDevFldCst[0].gridRes)
	{
		ptcGridIdx[i] = gcIdx;
		gridPtcOffset[i] = atomicAdd(&gridPtcNum[gcIdx], 1);
	}
	else
	{
		ptcGridIdx[i] = GRID_UNDEF;
	}

}

__global__ void ComputeDensAndPressCUDA(
	flt3ptr pos,
	fltptr  press,
	PtcAttrPtr ptcAttr,
	intptr ptcGridIdx,
	intptr gridPtcNum,
	intptr gridOffset,
	intptr gridPtcIdx)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int pgIdx = ptcGridIdx[i];
	if (pgIdx == GRID_UNDEF) return;

	float simuScale = gDevFldCst[0].simuScale;
	float sr = gDevFldCst[0].sr;
	float lutKernel = gDevFldCst[0].lutKernel;
	float poly6kernel = gDevFldCst[0].poly6Kernel;

	float3 posI = pos[i];

	float densSum = gDevFldCst[0].densContrib;
	//densSum = 0.0;
	for (size_t nbr = 0; nbr < MAX_NBR_NUM; ++nbr)
	{
		int nbrIdx = pgIdx + gDevFldCst[0].nbrOffset[nbr];

		if (!IsGridIdxValid(nbrIdx, gDevFldCst[0].gridTotalNum)) continue;
		if (gridPtcNum[nbrIdx] == 0) continue;

		int nbrFirst = gridOffset[nbrIdx];
		int nbrLast = nbrFirst + gridPtcNum[nbrIdx];

		for (size_t cellIdx = nbrFirst; cellIdx < nbrLast; ++cellIdx)
		{
			int j = gridPtcIdx[cellIdx];
			if (i == j) continue;

			float3 posIJ = (posI - pos[j]) * simuScale;
			float dist = Distance(posIJ);

			if (dist <= sr)
			{
				++ptcAttr[i].nbrCnt;
				densSum += M4LutKernel(lutKernel, dist, sr);
				//densSum += Poly6Kernel(poly6kernel, dist, sr);
			}
		}

	}

	ptcAttr[i].dens = densSum * ptcAttr[i].mass;
	press[i] = Max(0.0, (ptcAttr[i].dens - gDevFldCst[0].restDens) * gDevFldCst[0].k);
}

__global__ void ComputeAccelerationCUDA(
	flt3ptr pos,
	flt3ptr vel,
	flt3ptr acc,
	fltptr  press,
	PtcAttrPtr ptcAttr,
	intptr ptcGridIdx,
	intptr gridPtcNum,
	intptr gridOffset,
	intptr gridPtcIdx)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int pgIdx = ptcGridIdx[i];
	if (pgIdx == GRID_UNDEF) return;

	float simuScale = gDevFldCst[0].simuScale;
	float sr = gDevFldCst[0].sr;
	float spikyKernel = gDevFldCst[0].spikyKernel;
	float laplaceKernel = gDevFldCst[0].laplaceKernel;

	float pressI = press[i];
	float3 posI = pos[i];
	float3 velI = vel[i];

	float3 accSum = make_float3(0.0, 0.0, 0.0);

	for (size_t nbr = 0; nbr < MAX_NBR_NUM; ++nbr)
	{
		int nbrIdx = pgIdx + gDevFldCst[0].nbrOffset[nbr];

		if (!IsGridIdxValid(nbrIdx, gDevFldCst[0].gridTotalNum)) continue;
		if (gridPtcNum[nbrIdx] == 0) continue;

		int nbrFirst = gridOffset[nbrIdx];
		int nbrLast = nbrFirst + gridPtcNum[nbrIdx];

		for (size_t cellIdx = nbrFirst; cellIdx < nbrLast; ++cellIdx)
		{
			int j = gridPtcIdx[cellIdx];
			if (i == j) continue;

			float3 posIJ = (posI - pos[j]) * simuScale;
			float dist = Distance(posIJ);

			if (dist <= sr)
			{
				float pressJ = press[j];
				float densJ = ptcAttr[j].dens;

				float3 accPress = SpikyKernel1stDervt(spikyKernel, posIJ, sr);
				accPress *= (pressI + pressJ) / (2 * densJ);

				accSum += accPress;

				float3 velJI = vel[j] - velI;
				float3 accVisc = velJI * LaplaceKernel2ndDervt(laplaceKernel, dist, sr);
				accVisc *= ptcAttr[j].visc / densJ;

				accSum += accVisc;
			}
		}

	}

	acc[i] = accSum * ptcAttr[i].mass / ptcAttr[i].dens + gDevFldCst[0].gravity;
}

__global__ void AdvanceParticleCUDA(
	flt3ptr pos,
	flt3ptr vel,
	flt3ptr acc)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	float simuScale = gDevFldCst[0].simuScale;
	float timeStep = gDevFldCst[0].timeStep;

	pos[i] *= simuScale;

	vel[i] += acc[i] * timeStep;
	pos[i] += vel[i] * timeStep;

	BoundaryHandle(gDevFldCst[0].scene, pos[i] / simuScale, vel[i]);

	pos[i] /= simuScale;

}

__device__ void BoundaryHandle(int scene, float3& pos, float3& vel)
{
	switch (scene)
	{
	case 0:  CubeCollision(pos, vel); break;
	default: CubeCollision(pos, vel);
	}
}

__device__ void CubeCollision(float3& pos, float3& vel)
{
	float damping = 0.1;
	float reflect = 2.0;

	float r = gDevFldCst[0].ptcRadius;
	
	float3 boundMin = gDevFldCst[0].gridMin + make_float3(r, r, r);
	float3 boundMax = gDevFldCst[0].gridMax - make_float3(r, r, r);

	if (pos.x < boundMin.x)
	{
		pos.x = boundMin.x;
		float3 rn = make_float3(1.0, 0.0, 0.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.x *= damping;
	}
	if (pos.x > boundMax.x)
	{
		pos.x = boundMax.x;
		float3 rn = make_float3(-1.0, 0.0, 0.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.x *= damping;
	}
	if (pos.y < boundMin.y)
	{
		pos.y = boundMin.y;
		float3 rn = make_float3(0.0, 1.0, 0.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.y *= damping;
	}
	if (pos.y > boundMax.y)
	{
		pos.y = boundMax.y;
		float3 rn = make_float3(0.0, -1.0, 0.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.y *= damping;
	}
	if (pos.z < boundMin.z)
	{
		pos.z = boundMin.z;
		float3 rn = make_float3(0.0, 0.0, 1.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.z *= damping;
	}
	if (pos.z > boundMax.z)
	{
		pos.z = boundMax.z;
		float3 rn = make_float3(0.0, 0.0, -1.0);
		vel -= rn * Dot(rn, vel) * reflect;
		vel.z *= damping;
	}

}

__global__ void PBF_PredictPositionCUDA(
	flt3ptr pos,
	flt3ptr interPos,
	flt3ptr vel,
	flt3ptr acc)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	float simuScale = gDevFldCst[0].simuScale;
	float timeStep = gDevFldCst[0].timeStep;

	acc[i] = make_float3(0.0, -9.81, 0.0);
	vel[i] += acc[i] * timeStep;
	interPos[i] = pos[i] * simuScale + vel[i] * timeStep;

	// PBF在更新位置后才进行碰撞检测
	//BoundaryHandle(gDevFldCst[0].scene, pos[i] / simuScale, vel[i]);
}

__global__ void PBF_ComputeDensAndFactorCUDA(
	flt3ptr interPos,
	fltptr  factor,
	PtcAttrPtr ptcAttr,
	intptr ptcGridIdx,
	intptr gridPtcNum,
	intptr gridOffset,
	intptr gridPtcIdx)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	ptcAttr[i].nbrCnt = 0;
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int pgIdx = ptcGridIdx[i];
	if (pgIdx == GRID_UNDEF) return;

	float simuScale = gDevFldCst[0].simuScale;
	float sr = gDevFldCst[0].sr;
	float lutKernel = gDevFldCst[0].lutKernel;
	float spikyKernel = gDevFldCst[0].spikyKernel;
	float poly6Kernel = gDevFldCst[0].poly6Kernel;
	float restDens = gDevFldCst[0].restDens;
	float mass = gDevFldCst[0].ptcMass;

	float3 posI = interPos[i];

	float densSum = gDevFldCst[0].densContrib;
	//densSum = 0.0;

	float3 cGrad = make_float3(0.0, 0.0, 0.0);
	float  gradSum = 0.0;
	
	for (size_t nbr = 0; nbr < MAX_NBR_NUM; ++nbr)
	{
		int nbrIdx = pgIdx + gDevFldCst[0].nbrOffset[nbr];

		if (!IsGridIdxValid(nbrIdx, gDevFldCst[0].gridTotalNum)) continue;
		if (gridPtcNum[nbrIdx] == 0) continue;

		int nbrFirst = gridOffset[nbrIdx];
		int nbrLast = nbrFirst + gridPtcNum[nbrIdx];

		for (size_t cellIdx = nbrFirst; cellIdx < nbrLast; ++cellIdx)
		{
			int j = gridPtcIdx[cellIdx];
			if (i == j) continue;

			float3 posIJ = (posI - interPos[j]);// *simuScale;
			float dist = Distance(posIJ);

			if (dist <= sr)
			{
				++ptcAttr[i].nbrCnt;
				densSum += M4LutKernel(lutKernel, dist, sr);
				//densSum += Poly6Kernel(poly6Kernel, dist, sr);

				float3 tmpGrad = SpikyKernel1stDervt(spikyKernel, posIJ, sr) * mass / restDens;
				cGrad += tmpGrad;
				gradSum += Dot(tmpGrad, tmpGrad);
			}
		}

	}

	gradSum += Dot(cGrad, cGrad);
	ptcAttr[i].dens = densSum * mass;

	float c = ptcAttr[i].dens / restDens - 1.0f;
	
	c = c >= 0.0f ? c : 0.0;

	factor[i] = -c / (gradSum + gDevFldCst[0].relaxation);
}

__global__ void PBF_ComputePositionDeltaCUDA(
	flt3ptr interPos,
	flt3ptr vel,
	fltptr  factor,
	PtcAttrPtr ptcAttr,
	intptr ptcGridIdx,
	intptr gridPtcNum,
	intptr gridOffset,
	intptr gridPtcIdx)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	int pgIdx = ptcGridIdx[i];
	if (pgIdx == GRID_UNDEF) return;

	float simuScale = gDevFldCst[0].simuScale;
	float sr = gDevFldCst[0].sr;
	float lutKernel = gDevFldCst[0].lutKernel;
	float spikyKernel = gDevFldCst[0].spikyKernel;
	float poly6Kernel = gDevFldCst[0].poly6Kernel;

	float3 posI = interPos[i];
	float3 posDeltaI = make_float3(0.0, 0.0, 0.0);
	float3 testPos = make_float3(0.0, 0.0, 0.0);
	float  factorI = factor[i];

	for (size_t nbr = 0; nbr < MAX_NBR_NUM; ++nbr)
	{
		int nbrIdx = pgIdx + gDevFldCst[0].nbrOffset[nbr];

		if (!IsGridIdxValid(nbrIdx, gDevFldCst[0].gridTotalNum)) continue;
		if (gridPtcNum[nbrIdx] == 0) continue;

		int nbrFirst = gridOffset[nbrIdx];
		int nbrLast = nbrFirst + gridPtcNum[nbrIdx];

		for (size_t cellIdx = nbrFirst; cellIdx < nbrLast; ++cellIdx)
		{
			int j = gridPtcIdx[cellIdx];
			if (i == j) continue;

			float3 posIJ = (posI - interPos[j]);// *simuScale;
			float dist = Distance(posIJ);

			if (dist <= sr)
			{
				float k = 0.001;
				float n = 4.0;
				float deltaQ = 0.0;
				float term = M4LutKernel(lutKernel, dist, sr) / M4LutKernel(lutKernel, 0.3 * sr, sr);
				//float term = Poly6Kernel(poly6Kernel, dist, sr) / Poly6Kernel(poly6Kernel, sr * 0.3, sr);
				float pressFactor = -k * term * term * term * term;
				posDeltaI += SpikyKernel1stDervt(spikyKernel, posIJ, sr) * (factorI + factor[j] + pressFactor);
				/*posDeltaI.x = SpikyKernel1stDervt(spikyKernel, posIJ, sr).x * (factorI + factor[j] + pressFactor);
				posDeltaI.y =  (factorI + factor[j] + pressFactor);
				posDeltaI.z = SpikyKernel1stDervt(spikyKernel, posIJ, sr).z * (factorI + factor[j] + pressFactor);*/
			}
		}

	}

	interPos[i] += posDeltaI * gDevFldCst[0].ptcMass / gDevFldCst[0].restDens;
	
	float r = gDevFldCst[0].ptcRadius;

	float3 boundMin = gDevFldCst[0].gridMin + make_float3(r, r, r);
	float3 boundMax = gDevFldCst[0].gridMax - make_float3(r, r, r);

}

__global__ void PBF_UpdateVelocityCUDA(
	flt3ptr pos,
	flt3ptr interPos,
	flt3ptr vel)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	vel[i] = (interPos[i] - pos[i] * gDevFldCst[0].simuScale)/ gDevFldCst[0].timeStep;
	pos[i] = interPos[i] / gDevFldCst[0].simuScale;

}

__global__ void PBF_XSPHAndVorticityCUDA()
{

}

__global__ void PBF_BoundaryConstraintCUDA(
	flt3ptr interPos,
	flt3ptr vel)
{
	uint i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= gDevFldCst[0].ptcTotalNum) return;

	/*if (interPos[i].y <= -2.0)
	{
		interPos[i].y = -2.0;
		vel[i].y = 0.0;
	}*/
}