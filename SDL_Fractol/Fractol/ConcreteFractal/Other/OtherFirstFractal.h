#pragma once
#include "../../Abstractions/AFractal.h"

class OtherFirstFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
public:
	OtherFirstFractal(int width, int height, TDrawFunction func);
	void Reset() override;
};
