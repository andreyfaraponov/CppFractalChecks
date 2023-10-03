#include "OtherSecondFractal.h"

#include <thread>
#include <vector>

OtherSecondFractal::OtherSecondFractal(int width, int height, TDrawFunction func) : AFractal(width, height, func)
{
}

float OtherSecondFractal::GetI(const TPosition& pos) const
{
		TFractVals vals;
		int i;
		double a;
		double b;
		double temp;
	
		i = -1;
		vals.lastre = 4 * (pos.x - _width / 2) / (_zoom * _width) + _moveX;
		vals.lastim = 4 * (pos.y - _height / 2) / (_zoom * _height) + _moveY;
		a = 0;
		b = 0;
	
		while (a * a + b * b < 4 && ++i < _iterations)
		{
			temp = a * a * a * a + b * b * b * b - 6 * a * a * b * b +
				vals.lastre;
			b = 4 * a * b * (a * a - b * b) + vals.lastim;
			a = temp;
		}
		return i;
}


