
#ifndef FLUID_SYSTEM_CUDA_CUH
#define FLUID_SYSTEM_CUDA_CUH
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>

#include "math_support.cuh"
#include "fluid_const.cuh"
#include "particle_attrib.cuh"
#include "grid_cell_host.cuh"
#include "smooth_kernel.cuh"

typedef int*       intptr;
typedef float*     fltptr;
typedef float3*    flt3ptr;
typedef PtcAttrib* PtcAttrPtr;

// 以 gDev 为前缀的变量表示该数据为存储于GPU上的全局变量
// 以 gHost 为前缀的变量表示该数据为存储于CPU上的全局变量
static flt3ptr    gDevPos;           // 粒子的位置
extern flt3ptr    gHostPos;
static flt3ptr    gDevVel;           // 粒子的速度
extern flt3ptr    gHostVel;
static flt3ptr    gDevAcc;           // 粒子的加速度
static fltptr     gDevPress;         // 粒子的压强
static PtcAttrPtr gDevPtcAttr;       // 粒子的属性集合
extern PtcAttrPtr gHostPtcAttr;

static flt3ptr gDevPBFInterPos;
static fltptr  gDevPBFFactor;

//static intptr     gDevPtcGridIdx;    // 粒子所在单元格的索引
//static intptr     gDevGridPtcIdx;    // 单元格中粒子的索引
//static intptr     gDevGridPtcNum;    // 单元格所含有的粒子数目
//static intptr     gDevGridPtcOffset; // 单元格中粒子的偏移
//static intptr     gDevGridOffset;    // 单元格偏移

extern FluidConst gHostFldCst;
static __constant__ FluidConst gDevFldCst[1];

static int gBlockNum;  // 并行的区块数
static int gThreadNum; // 每个区块所执行的线程数

// 场景类型
enum Scene
{
	FLUID_IN_CUBE,            // 流动于立方体内的普通液体
	NON_NEWTON_FLUID_IN_CUBE, // 流动于立方体内的非牛顿流体
	BLOOD_IN_VESSEL           // 血管内的血液
};
static Scene gScene = Scene::FLUID_IN_CUBE;

void AdjustBlockAndThreadNum(int& blockNum, int& threadNum);

void PrintDeviceProperty();

// 分配全局数据
void AllocGlobalData(int ptcTotalNum, int gridTotalNum);

// 初始化全局数据
void InitGlobalData(int ptcTotalNum, int gridTotalNum);

// 释放全局数据
void FreeGlobalData();

void InsertParticles();

// 求前缀和
void PrefixSum();

// 计数排序
void CountingSort();

void FindNbrParticles();

// 初试化模拟配置
void InitSimulation(Scene scene);

void InitParticleAttrib(Scene scene, int ptcTotalNum);

// 构造搜索均匀网格，粒子必须在搜索网格内运动
void CreateGrid(
	Scene scene,
	const float3& gridMin,
	const float3& gridMax,
	float cellSize);

// 构造初试流体
void CreateFluid(
	Scene scene,
	const float3& fluidMin,
	const float3 & fluidMax,
	float spacing);

// 预计算核平滑函数常数项
void PreComputeKernels(FluidConst & fldCst);

// 预计算粒子间距
void PreComputeSpacing(FluidConst & fldCst);

// 预计算气体常数和模拟时间步长
void PreComputeGasConstantAndTimeStep(FluidConst & fldCst);

void FetchDataFromGPU();

void ComputeDensAndPress();

void ComputeAcceleration();

void AdvanceParticles();

void StandardSPH();

void PBF_PredictPosition();

void PBF_ComputeDensAndFactor();

void PBF_ComputePositionDelta();

void PBF_UpdateVelocity();

void PBF_XSPHAndVorticity();

void PBF_BoundaryConstraint();

void PBF();

void SimulateOnGPU();

void PrintCUDAError(const char* funcName);

__global__ void CountingSortCUDA(
	intptr ptcGridIdx = 0,
	intptr gridPtcOffset = 0,
	intptr gridOffset = 0,
	intptr gridPtcIdx = 0);

__global__ void InsertParticleCUDA(
	flt3ptr pos = 0,
	intptr  ptcGridIdx = 0,
	intptr  gridPtcNum = 0,
	intptr  gridPtcOffset = 0);

__global__ void ComputeDensAndPressCUDA(
	flt3ptr pos = 0,
	fltptr  press = 0,
	PtcAttrPtr ptcAttr = 0,
	intptr ptcGridIdx = 0,
	intptr gridPtcNum = 0,
	intptr gridOffset = 0,
	intptr girdPtcIdx = 0);

__global__ void ComputeAccelerationCUDA(
	flt3ptr pos = 0,
	flt3ptr vel = 0,
	flt3ptr acc = 0,
	fltptr  press = 0,
	PtcAttrPtr ptcAttr = 0,
	intptr ptcGridIdx = 0,
	intptr gridPtcNum = 0,
	intptr gridOffset = 0,
	intptr gridPtcIdx = 0);

__global__ void AdvanceParticleCUDA(
	flt3ptr pos = 0,
	flt3ptr vel = 0,
	flt3ptr acc = 0);

__device__ void BoundaryHandle(int scene, float3& pos, float3& vel);

__device__ void CubeCollision(float3& pos, float3& vel);

__global__ void PBF_PredictPositionCUDA(
	flt3ptr pos = 0,
	flt3ptr interPos = 0,
	flt3ptr vel = 0,
	flt3ptr acc = 0);

__global__ void PBF_ComputeDensAndFactorCUDA(
	flt3ptr interPos = 0,
	fltptr  factor = 0,
	PtcAttrPtr ptcAttr = 0,
	intptr ptcGridIdx = 0,
	intptr gridPtcNum = 0,
	intptr gridOffset = 0,
	intptr gridPtcIdx = 0);

__global__ void PBF_ComputePositionDeltaCUDA(
	flt3ptr interPos = 0,
	flt3ptr vel = 0,
	fltptr  factor = 0,
	PtcAttrPtr ptcAttr = 0,
	intptr ptcGridIdx = 0,
	intptr gridPtcNum = 0,
	intptr gridOffset = 0,
	intptr gridPtcIdx = 0);

__global__ void PBF_UpdateVelocityCUDA(
	flt3ptr pos = 0,
	flt3ptr interPos = 0,
	flt3ptr vel = 0);

__global__ void PBF_XSPHAndVorticityCUDA();

__global__ void PBF_BoundaryConstraintCUDA(
	flt3ptr interPos = 0,
	flt3ptr vel = 0);


#endif