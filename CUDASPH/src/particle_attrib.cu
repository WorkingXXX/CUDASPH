
#include "particle_attrib.cuh"

PtcAttrib CreateNormalParticle()
{
	PtcAttrib normalPtc;
	normalPtc.mass = 0.125;
	normalPtc.dens = 1.0;
	normalPtc.visc = 1.002;
	normalPtc.type = ParticleType::NORMAL;
	normalPtc.nbrCnt = 0;
	return normalPtc;
}

PtcAttrib CreateNonNewtonParticle()
{
	PtcAttrib nonNewtonPtc;
	return nonNewtonPtc;
}

PtcAttrib CreateRedCellParticle()
{
	PtcAttrib redCellPtc;
	return redCellPtc;
}

PtcAttrib CreatePlasmaParticle()
{
	PtcAttrib plasmaPtc;
	return plasmaPtc;
}

PtcAttrib CreateLargeParticle()
{
	PtcAttrib largePtc;
	largePtc.mass = 1.0;
	largePtc.dens = 0.0;
	largePtc.visc = 1.002;
	largePtc.type = ParticleType::LARGE;
	largePtc.nbrCnt = 0;
	return largePtc;
}