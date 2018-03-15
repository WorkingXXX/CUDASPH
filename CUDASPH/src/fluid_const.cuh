
#ifndef FLUID_CONST_CUH
#define FLUID_CONST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#define MAX_NBR_NUM 27

struct FluidConst
{
	int   ptcTotalNum; // 流体粒子总数
	float ptcMass;     // 粒子质量
	float restDens;    // 流体静态密度
	float densErrAllowed; // 所允许的最大密度误差
	float minVisc;     // 流体最小粘性系数
	float maxVisc;     // 流体最大粘性系数
	float k;           // 气体常数，用来计算压强
	float sr;          // 粒子平滑半径	
	float ptcRadius;   // 粒子图像上的半径
	float spaceGrpc;   // 粒子间的初始距离(图像上)
	float spaceReal;   // 粒子间的初始距离(实际上)
	float simuScale;   // 模拟规模缩放，用来放大距离以使得粒子可以显示
	float relaxation;
	float timeStep;    // 模拟的时间步长
	float poly6Kernel; // 核函数常数
	float spikyKernel;
	float laplaceKernel;
	float lutKernel;
	float densContrib;
	float cellDens;
	float cellSize;
	float3 gridMin; // 网格下限点
	float3 gridMax; // 网格上限点
	float3 gridSize; // 网格规格，即长宽高的大小
	int3 gridRes;   // 网格分辨率，即网格XYZ三个方向上的网格个数
	int    gridTotalNum; // 网格单元格的总个数
	float3 fluidInitMin; // 流体初始下限点
	float3 fluidInitMax; // 流体初始上限点
	float3 fluidInitSize;
	int3   fluidInitRes;
	float3 gravity;         // 重力加速度
	int    scene;
	int    nbrOffset[MAX_NBR_NUM];
};

// 初试化流体常数
void InitFluidConst(FluidConst & fldCst);



#endif