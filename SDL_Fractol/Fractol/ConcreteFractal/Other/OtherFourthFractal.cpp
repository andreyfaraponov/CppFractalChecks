#include "OtherFourthFractal.h"

OtherFourthFractal::OtherFourthFractal(int width, int height, TDrawFunction func) : AFractal(width, height, func)
{
}

float OtherFourthFractal::GetI(const TPosition& pos) const
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
	while (vals.newre * vals.newre + vals.newim * vals.newim <= 4 && ++i < _iterations)
	{
		x = vals.newre;
		y = vals.newim;
		vals.newre = x * x * x - 3 * x * y * y + vals.lastre;
		vals.newim = 3 * x * x * y - y * y * y + vals.lastim;
	}
	return i;
}
