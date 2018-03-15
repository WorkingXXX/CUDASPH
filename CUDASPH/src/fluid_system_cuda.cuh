
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

// �� gDev Ϊǰ׺�ı�����ʾ������Ϊ�洢��GPU�ϵ�ȫ�ֱ���
// �� gHost Ϊǰ׺�ı�����ʾ������Ϊ�洢��CPU�ϵ�ȫ�ֱ���
static flt3ptr    gDevPos;           // ���ӵ�λ��
extern flt3ptr    gHostPos;
static flt3ptr    gDevVel;           // ���ӵ��ٶ�
extern flt3ptr    gHostVel;
static flt3ptr    gDevAcc;           // ���ӵļ��ٶ�
static fltptr     gDevPress;         // ���ӵ�ѹǿ
static PtcAttrPtr gDevPtcAttr;       // ���ӵ����Լ���
extern PtcAttrPtr gHostPtcAttr;

static flt3ptr gDevPBFInterPos;
static fltptr  gDevPBFFactor;

//static intptr     gDevPtcGridIdx;    // �������ڵ�Ԫ�������
//static intptr     gDevGridPtcIdx;    // ��Ԫ�������ӵ�����
//static intptr     gDevGridPtcNum;    // ��Ԫ�������е�������Ŀ
//static intptr     gDevGridPtcOffset; // ��Ԫ�������ӵ�ƫ��
//static intptr     gDevGridOffset;    // ��Ԫ��ƫ��

extern FluidConst gHostFldCst;
static __constant__ FluidConst gDevFldCst[1];

static int gBlockNum;  // ���е�������
static int gThreadNum; // ÿ��������ִ�е��߳���

// ��������
enum Scene
{
	FLUID_IN_CUBE,            // �������������ڵ���ͨҺ��
	NON_NEWTON_FLUID_IN_CUBE, // �������������ڵķ�ţ������
	BLOOD_IN_VESSEL           // Ѫ���ڵ�ѪҺ
};
static Scene gScene = Scene::FLUID_IN_CUBE;

void AdjustBlockAndThreadNum(int& blockNum, int& threadNum);

void PrintDeviceProperty();

// ����ȫ������
void AllocGlobalData(int ptcTotalNum, int gridTotalNum);

// ��ʼ��ȫ������
void InitGlobalData(int ptcTotalNum, int gridTotalNum);

// �ͷ�ȫ������
void FreeGlobalData();

void InsertParticles();

// ��ǰ׺��
void PrefixSum();

// ��������
void CountingSort();

void FindNbrParticles();

// ���Ի�ģ������
void InitSimulation(Scene scene);

void InitParticleAttrib(Scene scene, int ptcTotalNum);

// �������������������ӱ����������������˶�
void CreateGrid(
	Scene scene,
	const float3& gridMin,
	const float3& gridMax,
	float cellSize);

// �����������
void CreateFluid(
	Scene scene,
	const float3& fluidMin,
	const float3 & fluidMax,
	float spacing);

// Ԥ�����ƽ������������
void PreComputeKernels(FluidConst & fldCst);

// Ԥ�������Ӽ��
void PreComputeSpacing(FluidConst & fldCst);

// Ԥ�������峣����ģ��ʱ�䲽��
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