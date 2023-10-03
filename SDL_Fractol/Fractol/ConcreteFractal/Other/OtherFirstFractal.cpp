#include "OtherFirstFractal.h"

OtherFirstFractal::OtherFirstFractal(int width, int height, TDrawFunction func) : AFractal(width, height, func)
{
}

float OtherFirstFractal::GetI(const TPosition& pos) const
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
		vals.newre = a * a - b * b + vals.lastre;
		vals.newim = -2 * a * b + vals.lastim;
	}
	
	return i;
}

void OtherFirstFractal::Reset()
{
	_zoom = 1;
	_moveX = 0;
	_moveY = 0;
	_iterations = 65;
}
