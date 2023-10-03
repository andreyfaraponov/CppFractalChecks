#include "OtherFifthFractal.h"

OtherFifthFractal::OtherFifthFractal(int width, int height, TDrawFunction func) : AFractal(width, height, func)
{
	_cre = -0.7176;
	_cim = -0.42511842;
}

float OtherFifthFractal::GetI(const TPosition& pos) const
{
	TFractVals vals;
	int i;
	double a;
	double b;

	i = -1;

	vals.newre = 1.5 * (pos.x - _width / 2) / (0.5 * _zoom * _width) + _moveX;
	vals.newim = 1.5 * (pos.y - _height / 2) / (0.5 * _zoom * _height) + _moveY;
	while ((vals.newre * vals.newre + vals.newim * vals.newim) <= 4 &&
		++i < _iterations)
	{
		a = vals.newre;
		b = vals.newim;
		vals.newre = a * a * a * a * a * a -
			15 * a * a * a * a * b * b + 15 * a * a * b * b * b * b
			- b * b * b * b * b * b - _currentCre;
		vals.newim = 6 * a * a * a * a * a * b -
			20 * a * a * a * b * b * b + 6 * a * b * b * b * b * b -
			_currentCim;
	}
	return i;
}

void OtherFifthFractal::DrawFractalPreconditions()
{
	_currentCre = _cre + _creAdd;
	_currentCim = _cim + _cimAdd;
}
