#pragma once
#include "../../Abstractions/AFractal.h"

class OtherSecondFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
public:
	OtherSecondFractal(int width, int height, TDrawFunction func);
};
