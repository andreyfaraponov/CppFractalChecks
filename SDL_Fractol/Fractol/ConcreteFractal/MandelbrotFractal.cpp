#include "MandelbrotFractal.h"
#include "cuda_runtime.h"

MandelbrotFractal::MandelbrotFractal(const int width, const int height, const TDrawFunction& func,
                                     const TDrawFunctionFromArray& funcFromArray) : AFractal(
	width, height, func)
{
	_arrayDrawFunction = funcFromArray;
	_pixelsArray = static_cast<unsigned char*>(malloc(sizeof(int) * width * height * 4));
	cudaSetDevice(0);
	cudaMalloc((void**)&_cudaPixelsArray, sizeof(int) * width * height * 4);
	_iterations = 65;
}

float MandelbrotFractal::GetI(const TPosition& pos) const
{
	TFractVals vals;
	int i;
	double x;
	double y;

	i = -1;

	vals.lastre = 4 * (pos.x - _width / 2) / (_zoom * _width) + _moveX;
	vals.lastim = 4 * (pos.y - _height / 2) / (_zoom * _height) + _moveY;
	vals.newre = 0;
	vals.newim = 0;

	while ((vals.newre * vals.newre + vals.newim * vals.newim) <= 4 &&
		++i < _iterations)
	{
		x = vals.newre;
		y = vals.newim;
		vals.newre = x * x - y * y + vals.lastre;
		vals.newim = 2 * x * y + vals.lastim;
	}

	double resultI = i;

	if (resultI < _iterations)
	{
		const double log_zn = log(x * x + y * y) / 2;
		const double nu = log(log_zn / log(2)) / log(2);
		resultI = resultI + 1.0 - nu;
	}

	return (resultI);
}

void MandelbrotFractal::Reset()
{
	_zoom = 1;
	_moveX = 0;
	_moveY = 0;
	_iterations = 65;
}

void MandelbrotFractal::DrawFractalSmooth()
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
	calculateOnCudaMandelbrot(data, _cudaPixelsArray);
	memset(_pixelsArray, 0, sizeof(int) * _width * _height * 4);
	cudaMemcpy(_pixelsArray, _cudaPixelsArray, sizeof(int) * _width * _height * 4, cudaMemcpyDeviceToHost);
	_arrayDrawFunction(_pixelsArray);
}
