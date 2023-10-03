#include <iostream>
#include <SDL.h>
#include <list>
#include "Fractol/PixelDrawer.h"
#include "Fractol\FractalDrawer.h"
#undef main

//PixelDrawer* drawer;
//FractalDrawer* fractalDrawer;
//bool isDisposed;

int X1 = 0;
int Y1 = 0;
int X2 = 0;
int Y2 = 0;
bool drawing = false;

struct Line
{
    int x1;
    int y1;
    int x2;
    int y2;
};

//void Dispose() {
//    if (isDisposed)
//        return;
//
//    isDisposed = true;
//
//    drawer->Close();
//
//    delete drawer;
//}

int main(int argc, char* argv[])
{
    PixelDrawer* drawer = new PixelDrawer();
    FractalDrawer* fractalDrawer = new FractalDrawer(drawer);

    drawer->Initialize();

    bool quit = false;
    SDL_Event e;
    
    while (!quit && !fractalDrawer->IsExit())
    {
        SDL_PollEvent(&e);

        switch (e.type)
        {
        case SDL_KEYDOWN:
            fractalDrawer->CheckInput(e.button.button);
            break;
        case SDL_MOUSEWHEEL:
            fractalDrawer->CheckZoom(e.wheel.preciseY);
            break;
        case SDL_QUIT:
            quit = true;
            break;
        }
        
        drawer->UpdateScreen();
    }

    std::cout << fractalDrawer->IsExit() << std::endl;

    drawer->Close();
    
    delete drawer;

    return 0;
}
