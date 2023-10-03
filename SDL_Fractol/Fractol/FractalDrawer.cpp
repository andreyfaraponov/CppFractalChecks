#include "FractalDrawer.h"

#include <iostream>
#include <thread>

#include "ConcreteFractal/JuliaFractal.h"
#include "ConcreteFractal/MandelbrotFractal.h"
#include "ConcreteFractal/Other/OtherFifthFractal.h"
#include "ConcreteFractal/Other/OtherFirstFractal.h"
#include "ConcreteFractal/Other/OtherFourthFractal.h"
#include "ConcreteFractal/Other/OtherSecondFractal.h"
#include "ConcreteFractal/Other/OtherSixthFractal.h"
#include "ConcreteFractal/Other/OtherThirdFractal.h"

FractalDrawer::FractalDrawer(PixelDrawer* pixel_drawer)
{
	_pixel_drawer = pixel_drawer;
	_exit = false;
}

void FractalDrawer::CheckInput(int key)
{
	auto drawPixelLambda = [=](int x, int y, RGBA rgb) { _pixel_drawer->DrawPixel(x, y, rgb); };
	auto drawPixelsFromArray = [=](unsigned char* pixels) { _pixel_drawer->DrawPixels(pixels); };
	switch (key)
	{
		case ESC:
			Exit();
			break;
		case ONE:
			delete _currentFractal;

			_currentFractal = new JuliaFractal(IMG_X, IMG_Y, drawPixelLambda, drawPixelsFromArray);
			DrawFractal();
			break;
		case TWO:
			delete _currentFractal;

			_currentFractal = new MandelbrotFractal(IMG_X, IMG_Y, drawPixelLambda, drawPixelsFromArray);
			DrawFractal();
			break;
		case THREE:
			delete _currentFractal;

			_currentFractal = new OtherFirstFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case FOUR:
			delete _currentFractal;

			_currentFractal = new OtherSecondFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case FIVE:
			delete _currentFractal;

			_currentFractal = new OtherThirdFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case SIX:
			delete _currentFractal;

			_currentFractal = new OtherFourthFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case SEVEN:
			delete _currentFractal;

			_currentFractal = new OtherFifthFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case EIGHT:
			delete _currentFractal;

			_currentFractal = new OtherSixthFractal(IMG_X, IMG_Y, drawPixelLambda);
			DrawFractal();
			break;
		case NINE:
			if (_currentFractal != nullptr)
				_currentFractal->ToggleSmoothFunction();

			DrawFractal();
			break;
		
		case MINUS:
			if (_currentFractal != nullptr)
			{
				_currentFractal->DecreaseIterations();
				DrawFractal();
			}
			break;
		case PLUS:
			if (_currentFractal != nullptr)
			{
				_currentFractal->IncreaseIterations();
				DrawFractal();
			}
			break;
		case LEFT:
			if (_currentFractal != nullptr)
			{
				_currentFractal->MoveLeft();
				DrawFractal();
			}
			break;
		case RIGHT:
			if (_currentFractal != nullptr)
			{
				_currentFractal->MoveRight();
				DrawFractal();
			}
			break;
		case UP:
			if (_currentFractal != nullptr)
			{
				_currentFractal->MoveUp();
				DrawFractal();
			}
			break;
		case DOWN:
			if (_currentFractal != nullptr)
			{
				_currentFractal->MoveDown();
				DrawFractal();
			}
			break;
		case RESET:
			Reset();
			break;
	}

	// key == COL_PLUS ? ft_col_up((t_mlx*)mlx) : NULL;
	// key == COL_MINUS ? ft_col_down((t_mlx*)mlx) : NULL;
	// key == PSYHO ? ft_psyho_mode((t_mlx*)mlx) : NULL;
}

void FractalDrawer::Reset() const
{
	if (_currentFractal != nullptr)
	{
		_currentFractal->Reset();
		DrawFractal();
	}
}

void FractalDrawer::CheckZoom(float precise_y) const
{
	if (precise_y > 0 && _currentFractal != nullptr)
		_currentFractal->ZoomPlus();
	else if (precise_y < 0 && _currentFractal != nullptr)
		_currentFractal->ZoomMinus();

	DrawFractal();
}

void FractalDrawer::Exit()
{
	_exit = true;
}

bool FractalDrawer::IsExit() const
{
	return _exit;
}

void FractalDrawer::DrawFractal() const
{
	if (_currentFractal != nullptr)
		_currentFractal->DrawFractal();
}
