
#ifndef PARTICLE_ATTRIB_CUH
#define PARTICLE_ATTRIB_CUH

enum ParticleType
{
	NORMAL,    // ��ͨ��������
	NONNEWTON, // ��ţ����������
	REDCELL,   // ��ϸ������
	PLASMA,    // Ѫ������
	LARGE,     // ��������
};

struct ParticleAttrib
{
	float mass;  // ����
	float dens;  // �ܶ�
	float visc;  // ճ��ϵ��
	int   type;  // ����
	int   nbrCnt;
};

typedef ParticleAttrib PtcAttrib;

PtcAttrib CreateNormalParticle();

PtcAttrib CreateNonNewtonParticle();

PtcAttrib CreateRedCellParticle();

PtcAttrib CreatePlasmaParticle();

PtcAttrib CreateLargeParticle();

#endif