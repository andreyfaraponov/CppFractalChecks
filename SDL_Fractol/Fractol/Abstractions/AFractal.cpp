#include "AFractal.h"

#include <iostream>
#include <valarray>

AFractal::AFractal(int width, int height, TDrawFunction drawPixel)
{
	_width = width;
	_height = height;
	_drawPixelFunction = drawPixel;
	_unit = 0.3 * width;
}
//
// double fc(double x)
// {
// 	// makes the color slightly smoother
// 	if (x <= 0.0) return (0.0);
// 	if (x >= 1.0) return (1.0);
// 	return (2.0 * (1.0 + x * sqrt(x) - (1 - x) * sqrt(1 - x)) - 3.0 * x);
// }
//
// void color(double x, double arr[])
// {
// 	// arbitrary phase shift
// 	x += 0.52;
// 	x -= (int)x;
//
// 	if (x < 0)
// 		x += 1.0;
// 	x *= 6;
// 	int xx = x;
// 	x -= xx;
//
// 	switch (xx)
// 	{
// 		case 0:
// 			arr[0] = 1.0;
// 			arr[1] = fc(1.0 - x);
// 			arr[2] = 0;
// 			break;
// 		case 1:
// 			arr[0] = 1.0;
// 			arr[1] = 0;
// 			arr[2] = fc(x);
// 			break;
// 		case 2:
// 			arr[0] = fc(1.0 - x);
// 			arr[1] = 0;
// 			arr[2] = 1.0;
// 			break;
// 		case 3:
// 			arr[0] = 0;
// 			arr[1] = fc(x);
// 			arr[2] = 1.0;
// 			break;
// 		case 4:
// 			arr[0] = 0;
// 			arr[1] = 1.0;
// 			arr[2] = fc(1.0 - x);
// 			break;
// 		case 5:
// 			arr[0] = fc(x);
// 			arr[1] = 1.0;
// 			arr[2] = 0;
// 			break;
// 	}
// }

// double flatten(double x0, double y0, double x, double y)
// {
// 	// creates a continuous gradient instead of color steps
// 	int steps = 0;
// 	double sqrxy = x * x + y * y;
// 	if (sqrxy >= 4.0)
// 	{
// 		while (sqrxy < 65536) // high value >> 2
// 		{
// 			double x_old = x;
// 			x = x * x - y * y;
// 			y = 2 * x_old * y;
// 			x += x0;
// 			y += y0;
// 			steps++;
// 			sqrxy = x * x + y * y;
// 		}
// 		double m = steps + 1.5 - log2(log2(sqrxy));
// 		return (m);
// 	}
// 	return (0.0);
// }

void AFractal::DrawFractalByThreads()
{
	const int processor_count = std::thread::hardware_concurrency();
	int width = _width / processor_count;
	std::vector<std::thread> threads{};

	for (int i = 0; i < processor_count; i++)
	{
		TPosition chunk;
		chunk.x = width * i;
		chunk.y = 0;
		chunk.img_x = _width;
		chunk.img_y = _height;
		chunk.width = width;
		threads.push_back(std::thread(&AFractal::DrawConcreteFractal, this, chunk));
	}

	for (int i = 0; i < processor_count; ++i)
		threads[i].join();
}

void AFractal::DrawConcreteFractal(TPosition chunk) const
{
	chunk.width = chunk.width + chunk.x;

	for (int x = chunk.x; x < chunk.width; ++x)
	{
		for (int y = 0; y < chunk.img_y; ++y)
		{
			TPosition pos;
			pos.x = x;
			pos.y = y;
			pos.img_x = chunk.img_x;
			pos.img_y = chunk.img_y;
			float i = GetI(pos);

			RGBA rgba;
			
			if (i <= 0)
			{
				rgba.r = 0;
				rgba.g = 0;
				rgba.b = 0;
			}
			else
			{
				RGBA rgb1 = GetColor(floor(i));
				RGBA rgb2 = GetColor(floor(i) + 1);
				
				rgba.r = Linear(rgb1.r, rgb2.r, fmod(i, 1));
				rgba.g = Linear(rgb1.g, rgb2.g, fmod(i, 1));
				rgba.b = Linear(rgb1.b, rgb2.b, fmod(i, 1));
			}
			rgba.a = 255;
			
			if (_drawPixelFunction != nullptr)
				_drawPixelFunction(x, y, rgba);
		}
	}
}

RGBA AFractal::GetColor(float iteration) const
{
	RGBA result;
	
	result.r = static_cast<unsigned char>(127.5 * (cos(iteration) + 1) + .5);
	result.g = static_cast<unsigned char>(127.5 * (sin(iteration) + 1) + .5);
	result.b = static_cast<unsigned char>(127.5 * (1 - cos(iteration)) + .5);
	result.a = 255;
	
	return result;
}

unsigned char AFractal::Linear(unsigned char a, unsigned char b, double f)
{
	return a * (1.0 - f) + (b * f);
}

void AFractal::DrawFractalPreconditions()
{
}

void AFractal::DrawFractal()
{
	DrawFractalPreconditions();

	if (_smooth)
	{
		DrawFractalSmooth();
	}
	else
	{
		DrawFractalByThreads();
	}
}

void AFractal::ToggleSmoothFunction()
{
	_smooth = !_smooth;
	Reset();
}

void AFractal::DrawFractalSmooth()
{
	std::cout << "Draw Smooth" << std::endl;
}

void AFractal::Reset()
{
	_zoom = 1;
	_moveX = 0;
	_moveY = 0;
	_iterations = 65;
	_creAdd = 0;
	_cimAdd = 0;
}

AFractal::~AFractal()
{
}

void AFractal::ZoomMinus()
{
	_zoom /= 1.1f;
}

void AFractal::ZoomPlus()
{
	_zoom *= 1.1f;
}

void AFractal::MoveLeft()
{
	_moveX -= 0.3 / _zoom;
}

void AFractal::MoveRight()
{
	_moveX += 0.3 / _zoom;
}

void AFractal::MoveUp()
{
	_moveY -= 0.3 / _zoom;
}

void AFractal::MoveDown()
{
	_moveY += 0.3 / _zoom;
}

void AFractal::IncreaseIterations()
{
	//_iterations++;
	_creAdd += 0.01;
	_cimAdd += 0.01;
}

void AFractal::DecreaseIterations()
{
	_creAdd -= 0.01;
	_cimAdd -= 0.01;
	//_iterations--;
}
