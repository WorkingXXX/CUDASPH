
#ifndef PARTICLE_ATTRIB_CUH
#define PARTICLE_ATTRIB_CUH

enum ParticleType
{
	NORMAL,    // 普通流体粒子
	NONNEWTON, // 非牛顿流体粒子
	REDCELL,   // 红细胞粒子
	PLASMA,    // 血浆粒子
	LARGE,     // 大型粒子
};

struct ParticleAttrib
{
	float mass;  // 质量
	float dens;  // 密度
	float visc;  // 粘性系数
	int   type;  // 种类
	int   nbrCnt;
};

typedef ParticleAttrib PtcAttrib;

PtcAttrib CreateNormalParticle();

PtcAttrib CreateNonNewtonParticle();

PtcAttrib CreateRedCellParticle();

PtcAttrib CreatePlasmaParticle();

PtcAttrib CreateLargeParticle();

#endif