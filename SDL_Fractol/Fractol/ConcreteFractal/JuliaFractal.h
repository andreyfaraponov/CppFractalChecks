#pragma once
#include "../Abstractions/AFractal.h"

extern "C" {
extern void calculateOnCudaJulia(CudaMandelbrotStruct data, unsigned char* pixelsResult);
}

class JuliaFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;
	void DrawFractalPreconditions() override;

	unsigned char* _cudaPixelsArray;
	unsigned char* _pixelsArray;
	TDrawFunctionFromArray _arrayDrawFunction;

public:
	JuliaFractal(int width, int height, TDrawFunction func, TDrawFunctionFromArray funcFromArray);
	void Reset() override;
	void DrawFractalSmooth() override;
};
