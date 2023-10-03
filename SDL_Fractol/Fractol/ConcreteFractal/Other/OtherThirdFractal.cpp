#include "OtherThirdFractal.h"

OtherThirdFractal::OtherThirdFractal(int width, int height, TDrawFunction func) : AFractal(width, height, func)
{
}

void OtherThirdFractal::DrawFractalPreconditions()
{
	_currentCim = _cim + _cimAdd;
	_currentCre = _cre + _creAdd;
}

float OtherThirdFractal::GetI(const TPosition& pos) const
{
	TFractVals vals;
	int i;
	double a;
	double b;

	i = -1;
	vals.lastre = 4 * (pos.x - _width / 2) / (_zoom * _width) + _moveX;
	vals.lastim = 4 * (pos.y - _height / 2) / (_zoom * _height) + _moveY;
	vals.newre = 0;
	vals.newim = 0;
	while (vals.newre * vals.newre + vals.newim * vals.newim <= 4 && ++i < _iterations)
	{
		a = vals.newre;
		b = vals.newim;
		vals.newre = -a * b * 2.0 + vals.lastre;
		vals.newim = fabs(a * a - b * b) + vals.lastim;
	}
	return i;
}
