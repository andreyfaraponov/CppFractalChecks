#pragma once
#include "../Abstractions/AFractal.h"

extern "C" {
extern void calculateOnCudaMandelbrot(CudaMandelbrotStruct data, unsigned char* pixelsResult);
}

class MandelbrotFractal : public AFractal
{
private:
	float GetI(const TPosition& pos) const override;

	unsigned char* _cudaPixelsArray;
	unsigned char* _pixelsArray;
	TDrawFunctionFromArray _arrayDrawFunction;

public:
	MandelbrotFractal(int width, int height, const TDrawFunction& func, const TDrawFunctionFromArray& funcFromArray);
	void DrawFractalSmooth() override;
	void Reset() override;
};
