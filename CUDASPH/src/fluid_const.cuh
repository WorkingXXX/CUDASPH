
#ifndef FLUID_CONST_CUH
#define FLUID_CONST_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#define MAX_NBR_NUM 27

struct FluidConst
{
	int   ptcTotalNum; // ������������
	float ptcMass;     // ��������
	float restDens;    // ���徲̬�ܶ�
	float densErrAllowed; // �����������ܶ����
	float minVisc;     // ������Сճ��ϵ��
	float maxVisc;     // �������ճ��ϵ��
	float k;           // ���峣������������ѹǿ
	float sr;          // ����ƽ���뾶	
	float ptcRadius;   // ����ͼ���ϵİ뾶
	float spaceGrpc;   // ���Ӽ�ĳ�ʼ����(ͼ����)
	float spaceReal;   // ���Ӽ�ĳ�ʼ����(ʵ����)
	float simuScale;   // ģ���ģ���ţ������Ŵ������ʹ�����ӿ�����ʾ
	float relaxation;
	float timeStep;    // ģ���ʱ�䲽��
	float poly6Kernel; // �˺�������
	float spikyKernel;
	float laplaceKernel;
	float lutKernel;
	float densContrib;
	float cellDens;
	float cellSize;
	float3 gridMin; // �������޵�
	float3 gridMax; // �������޵�
	float3 gridSize; // �����񣬼�����ߵĴ�С
	int3 gridRes;   // ����ֱ��ʣ�������XYZ���������ϵ��������
	int    gridTotalNum; // ����Ԫ����ܸ���
	float3 fluidInitMin; // �����ʼ���޵�
	float3 fluidInitMax; // �����ʼ���޵�
	float3 fluidInitSize;
	int3   fluidInitRes;
	float3 gravity;         // �������ٶ�
	int    scene;
	int    nbrOffset[MAX_NBR_NUM];
};

// ���Ի����峣��
void InitFluidConst(FluidConst & fldCst);



#endif