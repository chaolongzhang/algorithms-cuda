/*
* recursive gaussian filter by Young & Van Vliet 
* about recursive gaussian filter, see paper 
* Young I T, Van Vliet L J. Recursive implementation of the Gaussian filter[J]. Signal processing, 1995, 44(2): 139-151.
*/
#ifndef RGF_H
#define RGF_H

#include "RGF_kernels.cuh"

float with_to_sigma(float width);

void calc_coeff(float sigma, float &B, float &b0, float &b1, float &b2, float &b3);

int iDivUp(int a, int b);

template<typename T>
void transpose(T *d_dest, T *d_src, size_t width, size_t height)
{
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose<<< grid, threads >>>(d_dest, d_src, width, height);
}

template<typename T>
void yvrg_2d(T *out, T *in, size_t width, size_t height, float sigma = 0.8)
{
    float B, b0, b1, b2, b3;
    calc_coeff(sigma, B, b0, b1, b2, b3);

    yvrg_2d(out, in, width, height, B, b0, b1, b2, b3);
}

template<typename T>
void yvrg_2d(T *out, T *in, size_t width, size_t height, float B, float b0, float b1, float b2, float b3)
{
    T *d_out = NULL;
    T *d_in = NULL;
    T *d_w = NULL;

    size_t mem_size = sizeof(T) * width * height;

    cudaMalloc(&d_out, mem_size);
    cudaMalloc(&d_in, mem_size);
    cudaMalloc(&d_w, mem_size);

    // copy data to GPU
    cudaMemcpy(d_in, in, mem_size, cudaMemcpyHostToDevice);

    dim3 block(256, 1, 1);
    dim3 grid(iDivUp(height, 256), 1, 1);

    d_forward_pass<<<grid, block>>>(d_w, d_in, width, height, B, b0, b1, b2, b3);
    transpose(d_out, d_w, width, height);

    grid.x = iDivUp(width, 256);
    d_backward_pass<<<grid, block>>>(d_w, d_out, height, width, B, b0, b1, b2, b3);
    transpose(d_out, d_w, height, width);

    // copy data back to CPU
    cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_in);
    cudaFree(d_out);
}

#endif