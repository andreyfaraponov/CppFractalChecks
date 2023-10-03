#include "JuliaFractal.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

JuliaFractal::JuliaFractal(const int width, const int height, const TDrawFunction func,
                           const TDrawFunctionFromArray funcFromArray)
	: AFractal(width, height, func)
{
	_cim = -0.31111842;
	_cre = -0.7176;
	_iterations = 65;

	_arrayDrawFunction = funcFromArray;
	_pixelsArray = static_cast<unsigned char*>(malloc(sizeof(int) * width * height * 4));
	cudaSetDevice(0);
	cudaMalloc((void**)&_cudaPixelsArray, sizeof(int) * width * height * 4);
}

void JuliaFractal::DrawFractalPreconditions()
{
	_currentCre = _cre + _creAdd;
	_currentCim = _cim + _cimAdd;
}

float JuliaFractal::GetI(const TPosition& pos) const
{
	int i = -1;

	double x = 1.75 * (pos.x - _width * .5) / (0.5 * _zoom * _width) + _moveX;
	double y = 1.75 * (pos.y - _height * .5) / (0.5 * _zoom * _height) + _moveY;

	while (x * x + y * y <= 4 && ++i < _iterations)
	{
		double xtemp = x * x - y * y;
		y = 2 * x * y + _currentCim;
		x = xtemp + _currentCre;
	}

	double resultI = i;

	if (resultI < _iterations)
	{
		const double log_zn = log(x * x + y * y) / 2;
		const double nu = log(log_zn / log(2)) / log(2);
		resultI = resultI + 1.0 - nu;
	}

	return resultI;
}

void JuliaFractal::Reset()
{
	_cim = -0.31111842;
	_cre = -0.7176;
	_zoom = 1;
	_moveX = 0;
	_moveY = 0;
	_iterations = 65;
	_creAdd = 0;
	_cimAdd = 0;
}

void JuliaFractal::DrawFractalSmooth()
{
	CudaMandelbrotStruct data;
	data.width = _width;
	data.height = _height;
	data.zoom = _zoom;
	data.moveX = _moveX;
	data.moveY = _moveY;
	data.maxIterations = _iterations;
	data.cimAdd = _cimAdd;
	data.creAdd = _creAdd;
	calculateOnCudaJulia(data, _cudaPixelsArray);
	memset(_pixelsArray, 0, sizeof(int) * _width * _height * 4);
	cudaMemcpy(_pixelsArray, _cudaPixelsArray, sizeof(int) * _width * _height * 4, cudaMemcpyDeviceToHost);
	_arrayDrawFunction(_pixelsArray);
}
