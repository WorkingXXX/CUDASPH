
#ifndef MATH_SUPPORT_CUH
#define MATH_SUPPORT_CUH

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#define uint unsigned int
#define usht unsigned short
#define uchr unsigned char
#define SIZE_INT sizeof(int)
#define SIZE_FLT sizeof(float)
#define SIZE_UINT sizeof(uint)
#define SIZE_USHT sizeof(usht)
#define SIZE_UCHR sizeof(uchr)
#define SIZE_FLT3 sizeof(float3)
#define SIZE_INT3 sizeof(int3)
#define Max(a, b) (((a)>(b))?(a):(b))
#define Min(a, b) (((a)<(b))?(a):(b))

#define GRAVITY 9.8
#define CFL_FACTOR 0.4
#define MY_PI 3.14159265358979323846

struct Matrix3x3
{
	float data[9];
};

typedef Matrix3x3 Mtrx3;

// 转置矩阵
__host__ __device__ Mtrx3 TransposeMtrx(const Mtrx3 & mtrx);

// 求矩阵的迹
__host__ __device__ float MtrxTrace(const Mtrx3 & mtrx);

// 初试化矩阵
__host__ __device__ void InitMtrx3(Mtrx3 & mtrx);

// 生成单位矩阵
__host__ __device__ Mtrx3 IdentityMtrx3();

// 克罗内克乘积
__host__ __device__ Mtrx3 Multiply(const float3 & v1, const float3 & v2, bool is_kronecker_product = false);

// 打印矩阵数据
void PrintMtrx3(const Mtrx3 & mtrx);

// 向量单位化
__host__ __device__ void Normalize(float3 & v);

// 向量点乘
__host__ __device__ float Dot(const float3 & v1, const float3 & v2);

// 计算两个点之间的距离
__host__ __device__ float Distance(const float3 & v1, const float3 & v2 = make_float3(0.0, 0.0, 0.0));

// 各种数据类型的运算符重载
__host__ __device__ float3 operator + (const float3 & v1, const float3 & v2);
__host__ __device__ float3 operator + (const float3 & v, const float f);
__host__ __device__ float3 operator - (const float3 & v1, const float3 & v2);
__host__ __device__ float3 operator - (const float3 & v, const float f);
__host__ __device__ float3 operator * (const float3 & v, float f);
__host__ __device__ float3 operator * (const int3 & v, float f);
__host__ __device__ float3 operator / (const float3 & v1, const float3 & v2);
__host__ __device__ float3 operator / (const int3 & v1, const float3 & v2);
__host__ __device__ float3 operator / (const float3 & v, float f);
__host__ __device__ void   operator += (float3 & v1, const float3 & v2);
__host__ __device__ void   operator -= (float3 & v1, const float3 & v2);
__host__ __device__ void   operator += (float3 & v, float f);
__host__ __device__ void   operator -= (float3 & v, float f);
__host__ __device__ void   operator *= (float3 & v, float f);
__host__ __device__ void   operator /= (float3 & v, float f);
__host__ __device__ float3 operator *  (const Mtrx3 & mtrx, const float3 & v);
__host__ __device__ Mtrx3 operator *  (const Mtrx3 & mtrx, float v);
__host__ __device__ Mtrx3 operator *  (float v, const Mtrx3 & mtrx);
__host__ __device__ Mtrx3 operator /  (const Mtrx3 & mtrx, float v);
__host__ __device__ Mtrx3 operator +  (const Mtrx3 & mtrx1, const Mtrx3 & mtrx2);
__host__ __device__ Mtrx3 operator *  (const Mtrx3 & mtrx1, const Mtrx3 & mtrx2);
__host__ __device__ int3  operator & (const int3 & v1, const int3 & v2);
__host__ __device__ int3  operator - (const int3 & v, int i);
__host__ __device__ int3  operator + (const int3 & v1, const int3 & v2);
__host__ __device__ bool  operator <= (const int3 & v1, const int3 & v2);
__host__ __device__ bool  operator < (const int3 & v1, const int3 & v2);
__host__ __device__ float3 operator - (const float3 & v);

float NormalRand();

inline __device__ Mtrx3 TransposeMtrx(const Mtrx3 & mtrx)
{

	Mtrx3 tran_mtrx;

	tran_mtrx.data[0] = mtrx.data[0];
	tran_mtrx.data[1] = mtrx.data[3];
	tran_mtrx.data[2] = mtrx.data[6];
	tran_mtrx.data[3] = mtrx.data[1];
	tran_mtrx.data[4] = mtrx.data[4];
	tran_mtrx.data[5] = mtrx.data[7];
	tran_mtrx.data[6] = mtrx.data[2];
	tran_mtrx.data[7] = mtrx.data[5];
	tran_mtrx.data[8] = mtrx.data[8];

	return tran_mtrx;
}

inline __device__ float MtrxTrace(const Mtrx3 & mtrx)
{
	return mtrx.data[0] + mtrx.data[4] + mtrx.data[8];
}

inline __device__ void InitMtrx3(Mtrx3 & mtrx)
{
	for (int i = 0; i < 9; i++)
	{
		mtrx.data[i] = 0.0;
	}
}

inline Mtrx3 IdentityMtrx3()
{
	Mtrx3 mtrx;
	mtrx.data[0] = 1.0;
	mtrx.data[4] = 1.0;
	mtrx.data[8] = 1.0;
	return mtrx;
}

// 计算时，v1为行向量（x1, y1, z1)，v2为列向量(x2, y2, z2)T
inline __host__ __device__ Mtrx3 Multiply(const float3 & v1, const float3 & v2, bool is_kronecker_product)
{
	Mtrx3 mtrx;

	if (is_kronecker_product)
	{
		mtrx.data[0] = v1.x * v2.x;
		mtrx.data[1] = v1.y * v2.x;
		mtrx.data[2] = v1.z * v2.x;
		mtrx.data[3] = v1.x * v2.y;
		mtrx.data[4] = v1.y * v2.y;
		mtrx.data[5] = v1.z * v2.y;
		mtrx.data[6] = v1.x * v2.z;
		mtrx.data[7] = v1.y * v2.z;
		mtrx.data[8] = v1.z * v2.z;
	}
	else
	{
		mtrx.data[0] = v1.x * v2.x;
		mtrx.data[1] = v1.x * v2.y;
		mtrx.data[2] = v1.x * v2.z;
		mtrx.data[3] = v1.y * v2.x;
		mtrx.data[4] = v1.y * v2.y;
		mtrx.data[5] = v1.y * v2.z;
		mtrx.data[6] = v1.z * v2.x;
		mtrx.data[7] = v1.z * v2.y;
		mtrx.data[8] = v1.z * v2.z;
	}

	return mtrx;
}

inline void PrintMtrx3(const Mtrx3 & mtrx)
{
	printf("%.6f, %.6f, %.6f\n", mtrx.data[0], mtrx.data[1], mtrx.data[2]);
	printf("%.6f, %.6f, %.6f\n", mtrx.data[3], mtrx.data[4], mtrx.data[5]);
	printf("%.6f, %.6f, %.6f\n", mtrx.data[6], mtrx.data[7], mtrx.data[8]);
}

inline __host__ __device__ void Normalize(float3 & v)
{
	float r_sq = v.x * v.x + v.y * v.y + v.z * v.z;
	if (r_sq > 0.0)
	{
		float r = sqrtf(r_sq);
		v /= r;
	}
}

inline __host__ __device__ float Dot(const float3 & v1, const float3 & v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __host__ __device__ float Distance(const float3 & v1, const float3 & v2)
{
	return sqrt((v1.x - v2.x) * (v1.x - v2.x) +
		(v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z));
}

inline __host__ __device__ float3 operator + (const float3 & v1, const float3 & v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
inline __host__ __device__ float3 operator + (const float3 & v, const float f)
{
	return make_float3(v.x + f, v.y + f, v.z + f);
}
inline __host__ __device__ float3 operator - (const float3 & v1, const float3 & v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline __host__ __device__ float3 operator - (const float3 & v, const float f)
{
	return make_float3(v.x - f, v.y - f, v.z - f);
}
inline __host__ __device__ float3 operator * (const float3 & v, float f)
{
	return make_float3(v.x * f, v.y * f, v.z * f);
}
inline __host__ __device__ float3 operator * (const int3 & v, float f)
{
	return make_float3(v.x * f, v.y * f, v.z * f);
}
inline __host__ __device__ float3 operator / (const float3 & v1, const float3 & v2)
{
	return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
inline __host__ __device__ float3 operator / (const int3 & v1, const float3 & v2)
{
	return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
inline __host__ __device__ float3 operator / (const float3 & v, float f)
{
	return make_float3(v.x / f, v.y / f, v.z / f);
}
inline __host__ __device__ void   operator += (float3 & v1, const float3 & v2)
{
	v1.x += v2.x; v1.y += v2.y; v1.z += v2.z;
}
inline __host__ __device__ void   operator -= (float3 & v1, const float3 & v2)
{
	v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z;
}
inline __host__ __device__ void   operator += (float3 & v, float f)
{
	v.x += f; v.y += f; v.z += f;
}
inline __host__ __device__ void   operator -= (float3 & v, float f)
{
	v.x -= f; v.y -= f; v.z -= f;
}
inline __host__ __device__ void operator *= (float3 & v, float f)
{
	v.x *= f; v.y *= f; v.z *= f;
}
inline __host__ __device__ void operator /= (float3 & v, float f)
{
	v.x /= f; v.y /= f; v.z /= f;
}
inline __host__ __device__ float3 operator * (const Mtrx3 & mtrx, const float3 & v)
{
	float x = mtrx.data[0] * v.x + mtrx.data[1] * v.y + mtrx.data[2] * v.z;
	float y = mtrx.data[3] * v.x + mtrx.data[4] * v.y + mtrx.data[5] * v.z;
	float z = mtrx.data[6] * v.x + mtrx.data[7] * v.y + mtrx.data[8] * v.z;
	return make_float3(x, y, z);
}
inline __host__ __device__ Mtrx3 operator * (const Mtrx3 & mtrx, float v)
{
	Mtrx3 result_mtrx;
	for (int i = 0; i < 9; i++)
		result_mtrx.data[i] = mtrx.data[i] * v;
	return result_mtrx;
}
inline __host__ __device__ Mtrx3 operator * (float v, const Mtrx3 & mtrx)
{
	Mtrx3 result_mtrx;
	for (int i = 0; i < 9; i++)
		result_mtrx.data[i] = mtrx.data[i] * v;
	return result_mtrx;
}
inline __host__ __device__ Mtrx3 operator /  (const Mtrx3 & mtrx, float v)
{
	Mtrx3 result_mtrx;
	for (int i = 0; i < 9; i++)
		result_mtrx.data[i] = mtrx.data[i] / v;
	return result_mtrx;
}
inline __host__ __device__ Mtrx3 operator +  (const Mtrx3 & mtrx1, const Mtrx3 & mtrx2)
{

	Mtrx3 sum_mtrx;

	for (int i = 0; i < 9; i++)
	{
		sum_mtrx.data[i] = mtrx1.data[i] + mtrx2.data[i];
	}

	return sum_mtrx;
}
inline __host__ __device__ Mtrx3 operator * (const Mtrx3 & mtrx1, const Mtrx3 & mtrx2)
{
	Mtrx3 result;
	for (int k = 0; k < 9; k++)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				result.data[k] = mtrx1.data[3 * i] * mtrx2.data[j]
					+ mtrx1.data[3 * i + 1] * mtrx2.data[j + 3]
					+ mtrx1.data[3 * i + 2] * mtrx2.data[j + 6];
			}
		}
	}
	return result;
}
inline __host__ __device__ void operator += (Mtrx3 & mtrx, const Mtrx3 & mtrx2)
{
	for (int i = 0; i < 9; i++)
		mtrx.data[i] += mtrx2.data[i];
}
inline __host__ __device__ int3 operator & (const int3 & v1, const int3 & v2)
{
	return make_int3(v1.x & v2.x, v1.y & v2.y, v1.z & v2.z);
}
inline __host__ __device__ int3 operator - (const int3 & v, int i)
{
	return make_int3(v.x - i, v.y - i, v.z - i);
}
inline __host__ __device__ int3 operator + (const int3 & v1, const int3 & v2)
{
	return make_int3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
inline __host__ __device__ bool operator <= (const int3 & v1, const int3 & v2)
{
	return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z);
}
inline __host__ __device__ bool operator < (const int3 & v1, const int3 & v2)
{
	return (v1.x < v2.x && v1.y < v2.y && v1.z < v2.z);
}
inline __host__ __device__ float3 operator - (const float3 & v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

inline float NormalRand()
{
	return rand() / (float)RAND_MAX;
}

#endif