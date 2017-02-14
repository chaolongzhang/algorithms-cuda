
#include <iostream>
#include <string>
#include <stdlib.h>

#include "RGF_cuda.cuh"
#include "helper.h"
#include "timeclock.h"

void benchmark(float sigma);
void test(float sigma, size_t size);
void test_with_img(float sigma);

using namespace std;

int main(int argc, char const *argv[])
{
    float sigma = 3.9;

    if (argc > 1)
    {
        string sg = argv[1];
        sigma = stof(sg);
    }
    
    test_with_img(sigma);
    // benchmark(sigma);
}

void benchmark(float sigma)
{
    srand(0);
    size_t size = 256;
    size_t max_size = 4096;

    while(size <= max_size)
    {
        test(sigma, size);
        size += 256;
    }
}

void test(float sigma, size_t size)
{
    float *img = NULL;
    size_t rows = size;
    size_t cols = size;

    random_img(&img, rows, cols);

    float *out = new float[rows * cols];
    GPU_Timeclock tc;

    tc.start();
    yvrg_2d(out, img, cols, rows, sigma);
    tc.record();

    delete [] out;

    cout << rows << " * " << cols << " : " << tc.elapsed_time() << "ms" << endl;
}

void test_with_img(float sigma)
{
    float *img = NULL;
    size_t rows = 0;
    size_t cols = 0;

    if(load_img(&img, rows, cols) == false)
    {
        cout << "Fail to loading image." << endl;
        return;
    }

    float *out = new float[rows * cols];
    yvrg_2d(out, img, cols, rows, sigma);

    show_image(out, rows, cols);
    show_all();

    delete [] out;
}