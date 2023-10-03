#pragma once
#include "../../Abstractions/AFractal.h"

class OtherFourthFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
	
public:
	OtherFourthFractal(int width, int height, TDrawFunction func);
};
