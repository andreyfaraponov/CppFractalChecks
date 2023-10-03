#pragma once
#include "../../Abstractions/AFractal.h"

class OtherSixthFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
	void DrawFractalPreconditions() override;
public:
	OtherSixthFractal(int width, int height, TDrawFunction func);
};
