
#include "fluid_const.cuh"

void InitFluidConst(FluidConst& fldCst)
{
	fldCst.ptcTotalNum = 0;
	fldCst.simuScale = 0.025;
	fldCst.ptcRadius = 0.025;
	fldCst.restDens = 1000.0;
	fldCst.spaceReal = fldCst.ptcRadius * 2;
	fldCst.spaceGrpc = fldCst.spaceReal / fldCst.simuScale;
	fldCst.sr = fldCst.spaceReal * 2;
	fldCst.ptcMass = fldCst.restDens * fldCst.spaceReal * fldCst.spaceReal * fldCst.spaceReal ;
	fldCst.cellSize = fldCst.sr;
	fldCst.relaxation = 0.000001;
	fldCst.densErrAllowed = 0.01;
	fldCst.minVisc = 2.0;
	fldCst.maxVisc = 100.0;
	fldCst.k = 0.0;
	fldCst.timeStep = 0.0;
	fldCst.poly6Kernel = 0.0;
	fldCst.spikyKernel = 0.0;
	fldCst.laplaceKernel = 0.0;
	fldCst.cellDens = 1.0;
	fldCst.gridMin = make_float3(0.0, 0.0, 0.0);
	fldCst.gridMax = make_float3(0.0, 0.0, 0.0);
	fldCst.gridSize = make_float3(0.0, 0.0, 0.0);
	fldCst.gridRes = make_int3(0, 0, 0);
	fldCst.gridTotalNum = 0;
	fldCst.fluidInitMin = make_float3(0.0, 0.0, 0.0);
	fldCst.fluidInitMax = make_float3(0.0, 0.0, 0.0);
	fldCst.fluidInitRes = make_int3(0, 0, 0);
	fldCst.gravity = make_float3(0.0, -9.8, 0.0);
}

void InitPBFFluidConst(FluidConst& fldCst)
{
	fldCst.ptcTotalNum = 0;
	fldCst.restDens = 1000.0;
	fldCst.densErrAllowed = 0.01;
	fldCst.minVisc = 2.0;
	fldCst.maxVisc = 100.0;
	fldCst.k = 0.0;
	fldCst.sr = 0.0;
	fldCst.ptcRadius = 0.8;
	fldCst.spaceGrpc = 0.0;
	fldCst.spaceReal = 0.0;
	fldCst.simuScale = 0.005;
	fldCst.timeStep = 0.0;
	fldCst.poly6Kernel = 0.0;
	fldCst.spikyKernel = 0.0;
	fldCst.laplaceKernel = 0.0;
	fldCst.cellDens = 1.0;
	fldCst.cellSize = 0;
	fldCst.relaxation = 600.0;
	fldCst.gridMin = make_float3(0.0, 0.0, 0.0);
	fldCst.gridMax = make_float3(0.0, 0.0, 0.0);
	fldCst.gridSize = make_float3(0.0, 0.0, 0.0);
	fldCst.gridRes = make_int3(0, 0, 0);
	fldCst.gridTotalNum = 0;
	fldCst.fluidInitMin = make_float3(0.0, 0.0, 0.0);
	fldCst.fluidInitMax = make_float3(0.0, 0.0, 0.0);
	fldCst.fluidInitRes = make_int3(0, 0, 0);
	fldCst.gravity = make_float3(0.0, -9.8, 0.0);
}