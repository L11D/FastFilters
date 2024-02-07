#pragma once
#include <CL/cl.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include<omp.h>
#include"SupportFunctions.h"
using namespace std;

void readKernelCode(string path, int kernelMaxsize, char*& kernelCode, size_t& kernelSize)
{
	FILE* fp;
	fp = fopen(path.c_str(), "r");
	kernelCode = (char*)malloc(kernelMaxsize);
	kernelSize = fread(kernelCode, 1, kernelMaxsize, fp);
	fclose(fp);
}

cl_kernel createKernel(string path, int kernelMaxsize, string kernelName, cl_context& context, cl_device_id& deviceID, cl_int& ret)
{
	cl_program program;
	cl_kernel kernel = NULL;
	size_t kernelSize;
	char* kernelCode;
	readKernelCode(path, kernelMaxsize, kernelCode, kernelSize);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, (const size_t*)&kernelSize, &ret);
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
	kernel = clCreateKernel(program, kernelName.c_str(), &ret);
	return kernel;
}

Image negativeWithOpenCL(Image image, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	auto start = CLOCK();

	cl_int ret;
	cl_uint retNumPlatforms, retNumDevices;
	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context context;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_mem pBuffer;

	ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
	ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);
	context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret);
	commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);
	kernel = createKernel("kernel.cl", 10000, "negative", context, deviceID, ret);
	pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &ret);

	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();

	//cout << "OpenCL init time: " << time << endl;

	start = CLOCK();
	ret = clEnqueueWriteBuffer(commandQueue, pBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, p, 0, NULL, NULL);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pBuffer);

	size_t globalWorkSize[1] = { size };
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(commandQueue, pBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, p, 0, NULL, NULL);

	time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	t = time;
	//cout << "OpenCL time: " << time << endl;

	return imageFromBytesRGB(p, width, height);
}

Image gaussianBlurWithOpenCL(Image image, int radius, double sigma, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	int coreSize = 2 * radius + 1;
	float* core = new float[coreSize * coreSize];
	buildCore(core, coreSize, sigma);

	int args[] = { width, height, radius, coreSize };

	auto start = CLOCK();

	cl_int ret;
	cl_uint retNumPlatforms, retNumDevices;
	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context context;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
	ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);
	context = clCreateContext(NULL, retNumDevices, &deviceID, NULL, NULL, &ret);
	commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);
	kernel = createKernel("kernel.cl", 20000, "gaussianBlur", context, deviceID, ret);

	cl_mem pBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &ret);
	cl_mem coreBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * coreSize * coreSize, NULL, &ret);
	cl_mem argsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, NULL, &ret);
	
	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	//cout << "OpenCL init time: " << time<< endl;

	start = CLOCK();
	ret = clEnqueueWriteBuffer(commandQueue, pBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, p, 0, NULL, NULL);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pBuffer);
	ret = clEnqueueWriteBuffer(commandQueue, coreBuffer, CL_TRUE, 0, sizeof(float) * coreSize * coreSize, core, 0, NULL, NULL);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &coreBuffer);
	ret = clEnqueueWriteBuffer(commandQueue, argsBuffer, CL_TRUE, 0, sizeof(int) * 4, args, 0, NULL, NULL);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &argsBuffer);

	size_t globalWorkSize[1] = { size/3 };
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(commandQueue, pBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, p, 0, NULL, NULL);
	
	time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	//cout << "OpenCL time: " << time << endl;
	t = time;
	clReleaseMemObject(pBuffer);
	clReleaseMemObject(coreBuffer);
	clReleaseMemObject(argsBuffer);

	return imageFromBytesRGB(p, width, height);
}