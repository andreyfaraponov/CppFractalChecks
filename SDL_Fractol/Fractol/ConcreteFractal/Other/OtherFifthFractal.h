#pragma once
#include "../../Abstractions/AFractal.h"

class OtherFifthFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
	void DrawFractalPreconditions() override;
public:
	OtherFifthFractal(int width, int height, TDrawFunction func);
};
