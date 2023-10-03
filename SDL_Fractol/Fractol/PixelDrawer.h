#pragma once
#include <SDL_render.h>
#include <SDL_video.h>

#include "Abstractions/AFractal.h"
#define ALPHA 255
# define W_X_SIZE 1200
# define W_Y_SIZE 1200
# define IMG_X 1024
# define IMG_Y 1024
# define W_NAME "Fractol"

class PixelDrawer
{
private:
    bool _isInitialized = false;
    SDL_Window* _window = nullptr;
    SDL_Renderer* _renderer = nullptr;
    SDL_Surface* _surface = nullptr;
    SDL_Texture* _texture;
    unsigned char _pixels[IMG_X * IMG_Y * 4];
    bool _isDirty;

public:
    void Initialize();
    void ClearScreen();
    void DrawPixel(int x, int y, RGBA);
    void DrawPixels(const unsigned char *pixels);
    void UpdateScreen();
    void Close();
};
