#pragma once
#include <functional>
#include <thread>
#include <cmath>
#include <vector>

const int subpix = 4;

typedef struct
{
	double newre;
	double newim;
	double lastre;
	double lastim;
} TFractVals;

typedef struct
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
} RGBA;

typedef struct
{
	int x;
	int y;
	int width;
	int img_x;
	int img_y;
} TPosition;

typedef struct
{
	int width;
	int height;
	int maxIterations;
	float zoom;
	float moveX;
	float moveY;
	float creAdd;
	float cimAdd;
} CudaMandelbrotStruct;

typedef std::function<void (int, int,RGBA)> TDrawFunction;
typedef std::function<void (unsigned char[])> TDrawFunctionFromArray;

class AFractal
{
protected:
	double _zoom = 1;
	double _moveX = 0;
	double _moveY = 0;
	double _creAdd = 0;
	double _cimAdd = 0;
	double _currentCre = 0;
	double _currentCim = 0;
	int _width = 0;
	int _height = 0;
	int _iterations = 40;
	double _cre = 0;
	double _cim = 0;
	double _unit = 0;
	TDrawFunction _drawPixelFunction;
	bool _smooth = false;

	AFractal(int width, int height, TDrawFunction drawPixel);
	
	static unsigned char Linear(unsigned char c, unsigned char r, double fmod);
	
	void DrawFractalByThreads();
	void DrawConcreteFractal(TPosition chunk) const;
	
	RGBA GetColor(float iteration) const;
	virtual void DrawFractalPreconditions();
	virtual float GetI(const TPosition& pos) const = 0;
	

public:
	void DrawFractal();
	void ToggleSmoothFunction();
	virtual void DrawFractalSmooth();
	virtual void Reset();
	virtual ~AFractal();

	
	virtual void ZoomMinus();
	virtual void ZoomPlus();
	virtual void MoveLeft();
	virtual void MoveRight();
	virtual void MoveUp();
	virtual void MoveDown();
	virtual void IncreaseIterations();
	virtual void DecreaseIterations();
};
