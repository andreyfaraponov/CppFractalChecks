#pragma once

# define GET_COLOR(r, g, b) ((r << 16) | (g << 8) | b)
# define GET_PSYHO(mlx, a, i) ((i * (mlx->col_counter + a)) % 255)
# define ISDYN(type) (type == 0)

# define ESC 41
# define LEFT 80
# define RIGHT 79
# define DOWN 81
# define UP 82
# define PLUS 87
# define MINUS 86
# define RESET 21
# define COL_PLUS 48
# define COL_MINUS 47
# define ONE 30
# define TWO 31
# define THREE 32
# define FOUR 33
# define FIVE 34
# define SIX 35
# define SEVEN 36
# define EIGHT 37
# define NINE 38

#include "PixelDrawer.h"
#include "Abstractions/AFractal.h"

class FractalDrawer
{
private:
    bool _exit = false;
    PixelDrawer* _pixel_drawer;

    AFractal* _currentFractal {nullptr};

    void Exit();
    void Reset() const;
public:
    FractalDrawer(PixelDrawer* pixel_drawer);
    void CheckInput(int button);
    bool IsExit() const;
    
    void DrawFractal() const;
    void CheckZoom(float precise_y) const;
};
