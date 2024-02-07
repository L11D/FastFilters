#pragma once
#include<iostream>
#include<intrin.h>
#include <immintrin.h>
#include "SupportFunctions.h"
#define CHAR_IN_MM256 32
#define SHORT_IN_MM256 16
#define INT_IN_MM256 8
using namespace std;

#pragma optimize("", off)
Image negativeWithIntrinsics(Image image, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	int mmSize = size / CHAR_IN_MM256;

	auto start = CLOCK();

	for (int i = 0; i < mmSize; i++)
	{
		__m256i Reg255 = _mm256_set1_epi8(255);
		__m256i RegP = _mm256_loadu_epi8(&p[i*CHAR_IN_MM256]);
		_mm256_storeu_epi8(&p[i* CHAR_IN_MM256], _mm256_sub_epi8(Reg255, RegP));
	}
	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	t = time;
	//cout << "negative with intrinsics: " << time;

	return imageFromBytesRGB(p, width, height);
}

#pragma optimize("", off)
Image gaussianBlurWithIntrinsics(Image image, int radius, float sigma, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	float* rs = new float[size / 3];
	float* gs = new float[size / 3];
	float* bs = new float[size / 3];

	int coreSize = 2 * radius + 1;
	float* core = new float[coreSize * coreSize];
	buildCore(core, coreSize, sigma);

	auto start = CLOCK();

	for (int i = 0, j = 0; i < size / 3; i++)
	{
		rs[i] = p[j++];
		gs[i] = p[j++];
		bs[i] = p[j++];
	}

	//#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x += INT_IN_MM256)
		{
			__m256 newRsReg = _mm256_setzero_ps();
			__m256 newGsReg = _mm256_setzero_ps();
			__m256 newBsReg = _mm256_setzero_ps();

			for (int j = 0; j < coreSize; j++)
			{
				if (y + j - radius < 0 || y + j - radius >= height) continue;
				for (int i = 0; i < coreSize; i++)
				{
					if (x + i - radius < 0 || x + i - radius >= height) continue;
					int index = (y + j) * width + x + i;
					if (index + 7 > width * height) continue;

					__m256 coreReg = _mm256_set1_ps(core[j * coreSize + i]);
					__m256 rReg = _mm256_loadu_ps(&rs[index]);
					__m256 gReg = _mm256_loadu_ps(&gs[index]);
					__m256 bReg = _mm256_loadu_ps(&bs[index]);

					rReg = _mm256_mul_ps(rReg, coreReg);
					gReg = _mm256_mul_ps(gReg, coreReg);
					bReg = _mm256_mul_ps(bReg, coreReg);

					newRsReg = _mm256_add_ps(newRsReg, rReg);
					newGsReg = _mm256_add_ps(newGsReg, gReg);
					newBsReg = _mm256_add_ps(newBsReg, bReg);

					/*coreReg = _mm256_setzero_ps();
					rReg = _mm256_setzero_ps();
					gReg = _mm256_setzero_ps();
					bReg = _mm256_setzero_ps();*/
				}
			}

			int index = 3 * (y * width + x);

			for (int i = 0; i < INT_IN_MM256; i++)
			{
				p[index + i*3] = ((float*)&newRsReg)[i];
				p[index + 1 + i*3] = ((float*)&newGsReg)[i];
				p[index + 2 + i*3] = ((float*)&newBsReg)[i];
			}

			/*newRsReg = _mm256_setzero_ps();
			newGsReg = _mm256_setzero_ps();
			newBsReg = _mm256_setzero_ps();*/
		}
	}

	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	t = time;
	//cout << "Consistent gaussian blur: " << time << endl;

	Image img = imageFromBytesRGB(p, width, height);
	//delete[]p, bs, rs, gs;
	return img;
}
