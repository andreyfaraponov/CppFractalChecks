#include "CuCalculations.cuh"

extern "C" {
__device__ double fc3(double x)
{
	// makes the color2 slightly smoother
	if (x <= 0.0)
		return 0.0;

	if (x >= 1.0)
		return 1.0;

	return 2.0 * (1.0 + x * sqrtf(x) - (1 - x) * sqrtf(1 - x)) - 3.0 * x;
}

__device__ void color2(double x, double arr[])
{
	// arbitrary phase shift
	x += 0.52;
	x -= (int)x;

	if (x < 0)
		x += 1.0;
	x *= 6;
	int xx = x;
	x -= xx;

	switch (xx)
	{
		case 0:
			arr[0] = 1.0;
			arr[1] = fc3(1.0 - x);
			arr[2] = 0;
			break;
		case 1:
			arr[0] = 1.0;
			arr[1] = 0;
			arr[2] = fc3(x);
			break;
		case 2:
			arr[0] = fc3(1.0 - x);
			arr[1] = 0;
			arr[2] = 1.0;
			break;
		case 3:
			arr[0] = 0;
			arr[1] = fc3(x);
			arr[2] = 1.0;
			break;
		case 4:
			arr[0] = 0;
			arr[1] = 1.0;
			arr[2] = fc3(1.0 - x);
			break;
		case 5:
			arr[0] = fc3(x);
			arr[1] = 1.0;
			arr[2] = 0;
			break;
	}
}

__device__ double flatten2(double x0, double y0, double x, double y)
{
	// creates a continuous gradient instead of color2 steps
	int steps = 0;
	double sqrxy = x * x + y * y;
	if (sqrxy >= 4.0)
	{
		while (sqrxy < 65536) // high value >> 2
		{
			double x_old = x;
			x = x * x - y * y;
			y = 2 * x_old * y;
			x += x0;
			y += y0;
			steps++;
			sqrxy = x * x + y * y;
		}

		return steps + 1.5 - log2f(log2f(sqrxy));
	}

	return 0.0;
}
	
__global__ void calculateSubPixel(int valX, int valY, const CudaMandelbrotStruct data, double* arr)
{
	const int i = static_cast<int>(blockIdx.x);
	const int j = static_cast<int>(threadIdx.x);

	double x0 = 4 * (valX - data.width * .5 - (j + 0.5) / subpix) / (data.zoom * data.width) + data.moveX;
	double y0 = 4 * (valY - data.height * .5 - (i + 0.5) / subpix) / (data.zoom * data.height) + data.moveY;
	// double x0 = (_width * mx - column - (j + 0.5) / subpix) / _unit;
	// double y0 = (_height * (1.0 - my) - row - (i + 0.5) / subpix) / _unit;
	double x = x0, y = y0;
	int n = -1;


	for (int it = 0; it <= data.maxIterations; it++)
	{
		// iteration steps
		/********* fractal generation ********/
		double x_old = x;
		x = x * x - y * y + x0;
		y = 2 * x_old * y + y0;
		/*************************************/
		if (x * x + y * y > 4)
		{
			// value is outside
			n = it;
			break;
		}
	}


	double subpixel[3];

	if (n < 0)
	{
		// black color2 inside
		for (int c = 0; c < 3; c++)
			subpixel[c] = 0;
	}
	else
	{
		double m = n;

		m += flatten2(x0, y0, x, y);
		if (m <= -1.8)
			m = -0.8;
		m = 0.23 * logf(1.8 + m);
		color2(m, subpixel);
	}

	for (int c = 0; c < 3; c++)
	{
		arr[c] += subpixel[c];
	}
}

__global__ void calculateMandelbrot(CudaMandelbrotStruct data, unsigned char* pixelsResult)
{
	const int valX = static_cast<int>(blockIdx.x);
	const int valY = static_cast<int>(threadIdx.x);

	data.maxIterations = 75;
	const double multiplier = 2 + data.creAdd;

	double sum[3] = {0.0, 0.0, 0.0};

	for (int i = 0; i < subpix; i++)
	{
		for (int j = 0; j < subpix; j++)
		{
			double x0 = 4 * (valX - data.width * .5 - (j + 0.5) / subpix) / (data.zoom * data.width) + data.moveX;
			double y0 = 4 * (valY - data.height * .5 - (i + 0.5) / subpix) / (data.zoom * data.height) + data.moveY;

			double x = x0, y = y0;
			int n = -1;

			for (int it = 0; it <= data.maxIterations; it++)
			{
				// iteration steps
				/********* fractal generation ********/
				double x_old = x;
				x = x * x - y * y + x0 + data.cimAdd;
				y = multiplier * x_old * y + y0;
				/*************************************/
				if (x * x + y * y > 4)
				{
					n = it;
					break;
				}
			}


			double subpixel[3];

			if (n < 0)
			{
				for (int c = 0; c < 3; c++)
					subpixel[c] = 0;
			}
			else
			{
				double m = n;

				m += flatten2(x0, y0, x, y);
				if (m <= -1.8)
					m = -0.8;
				m = 0.23 * logf(1.8 + m);
				color2(m, subpixel);
			}

			for (int c = 0; c < 3; c++)
			{
				sum[c] += subpixel[c];
			}
		}
	}
	
	char pixel[3];

	for (int c = 0; c < 3; c++)
	{
		pixel[c] = static_cast<int>(255.0 * sum[c] / (subpix * subpix) + 0.5);
	}

	RGBA rgba;
	rgba.r = pixel[0];
	rgba.g = pixel[1];
	rgba.b = pixel[2];
	rgba.a = 255;

	const unsigned int offset = data.width * valY * 4 + valX * 4;
	pixelsResult[offset + 0] = rgba.r;
	pixelsResult[offset + 1] = rgba.g;
	pixelsResult[offset + 2] = rgba.b;
	pixelsResult[offset + 3] = rgba.a;
}

__global__ void calculateJulia(CudaMandelbrotStruct data, unsigned char* pixelsResult)
{
	const int valX = static_cast<int>(blockIdx.x);
	const int valY = static_cast<int>(threadIdx.x);
	double _cim = -0.31111842 + data.cimAdd;
	double _cre = -0.7176 + data.creAdd;
	data.maxIterations = 365;

	int i = -1;

	double x = 1.75 * (valX - data.width * .5) / (0.5 * data.zoom * data.width) + data.moveX;
	double y = 1.75 * (valY - data.height * .5) / (0.5 * data.zoom * data.height) + data.moveY;

	while (x * x + y * y <= 4 && ++i < data.maxIterations)
	{
		double xtemp = x * x - y * y;
		y = 2 * x * y + _cim;
		x = xtemp + _cre;
	}

	double resultI = i;

	if (resultI < data.maxIterations)
	{
		const double log_zn = logf(x * x + y * y) / 2;
		const double nu = logf(log_zn / logf(2)) / logf(2);
		resultI = resultI + 1.0 - nu;
	}

	RGBA rgba;

	rgba.r = static_cast<unsigned char>(127.5 * (cosf(resultI) + 1) + .5);
	rgba.g = static_cast<unsigned char>(127.5 * (sinf(resultI) + 1) + .5);
	rgba.b = static_cast<unsigned char>(127.5 * (1 - cosf(resultI)) + .5);
	rgba.a = 255;

	const unsigned int offset = data.width * valY * 4 + valX * 4;
	pixelsResult[offset + 0] = rgba.r;
	pixelsResult[offset + 1] = rgba.g;
	pixelsResult[offset + 2] = rgba.b;
	pixelsResult[offset + 3] = rgba.a;
}

void calculateOnCudaJulia(CudaMandelbrotStruct data, unsigned char* pixelsResult)
{
	calculateJulia<<<data.width, data.height>>>(data, pixelsResult);
}

void calculateOnCudaMandelbrot(CudaMandelbrotStruct data, unsigned char* pixelsResult)
{
	calculateMandelbrot<<<data.width, data.height>>>(data, pixelsResult);
}
}