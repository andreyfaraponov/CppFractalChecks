#include "PixelDrawer.h"

#include <cstdio>
#include <iostream>
#include <SDL.h>

void PixelDrawer::Initialize()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL could not initialize! SDL Error: %s\n", SDL_GetError());
    }
    else
    {
        _window = SDL_CreateWindow(W_NAME,
                                   SDL_WINDOWPOS_CENTERED_MASK,
                                   SDL_WINDOWPOS_CENTERED_MASK,
                                   W_X_SIZE,
                                   W_Y_SIZE, SDL_WINDOW_OPENGL);

        if (_window == nullptr)
        {
            printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        }
        else
        {
            _renderer = SDL_CreateRenderer(_window, 0, 0);

            if (_renderer == nullptr)
            {
                printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
            }
            else
            {
                _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, IMG_X,
                                             IMG_Y);
                _isInitialized = true;
            }
        }
    }
}

void PixelDrawer::ClearScreen()
{
    if (!_isInitialized)
        return;

    for (int x = 0; x < IMG_X * IMG_Y * 4; ++x)
    {
        _pixels[x] = 0;
    }

    _isDirty = true;
}

void PixelDrawer::DrawPixel(int x, int y, RGBA rgba)
{
    _isDirty = true;
    const unsigned int offset = IMG_X * y * 4 + x * 4;
    _pixels[offset + 0] = rgba.r;
    _pixels[offset + 1] = rgba.g;
    _pixels[offset + 2] = rgba.b;
    _pixels[offset + 3] = rgba.a;
}

void PixelDrawer::DrawPixels(const unsigned char *pixels)
{
    SDL_UpdateTexture( _texture, nullptr, pixels, IMG_X * 4 );
    SDL_RenderCopy( _renderer, _texture, nullptr, nullptr );
    SDL_RenderPresent(_renderer);
    _isDirty = false;
}

void PixelDrawer::UpdateScreen()
{
    if (!_isInitialized || !_isDirty)
        return;
    
    SDL_UpdateTexture( _texture, nullptr, &_pixels, IMG_X * 4 );
    SDL_RenderCopy( _renderer, _texture, nullptr, nullptr );
    SDL_RenderPresent(_renderer);
    _isDirty = false;
}

void PixelDrawer::Close()
{
    SDL_DestroyWindow(_window);
    _window = nullptr;

    SDL_Quit();
}
