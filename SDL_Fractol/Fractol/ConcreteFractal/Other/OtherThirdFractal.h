#pragma once
#include "../../Abstractions//AFractal.h"

class OtherThirdFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
	void DrawFractalPreconditions() override;
public:
	OtherThirdFractal(int width, int height, TDrawFunction func);
};
