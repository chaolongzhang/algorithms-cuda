#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include "dft2d.cuh"

#define NUMS_COUNT_M (128)
#define NUMS_COUNT_N (128)
#define NUM_THREADS (32)

int main(int argc, char const *argv[])
{
    // M * N
    unsigned int numsCount = NUMS_COUNT_M * NUMS_COUNT_N;
    thrust::host_vector<float> h_vec(numsCount);
    thrust::host_vector<float> h_out_re(numsCount);
    thrust::host_vector<float> h_out_im(numsCount);
    thrust::host_vector<float> h_iout_re(numsCount);
    thrust::host_vector<float> h_iout_im(numsCount);

    thrust::device_vector<float> d_vec(numsCount);
    thrust::device_vector<float> d_out_re(numsCount);
    thrust::device_vector<float> d_out_im(numsCount);
    thrust::device_vector<float> d_iout_re(numsCount);
    thrust::device_vector<float> d_iout_im(numsCount);

    // fill h_vec with 2d array
    // [ [1, 1, ..., 1 ], [ 2, 2, ..., 2 ], ,,, ]
    for (int y = 0; y < NUMS_COUNT_M; ++y)
    {
        thrust::fill(h_vec.begin() + (y * NUMS_COUNT_N), h_vec.begin() + (y + 1) * NUMS_COUNT_N, (y + 1));
    }

    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
    thrust::fill(d_out_re.begin(), d_out_re.end(), 0);
    thrust::fill(d_out_im.begin(), d_out_im.end(), 0);

    int threads_xy = NUM_THREADS;
    int blocks_x = (NUM_THREADS + NUMS_COUNT_N - 1) / NUM_THREADS;
    int blocks_y = (NUM_THREADS + NUMS_COUNT_M - 1) / NUM_THREADS;

    std::cout << "blocks: ( " << blocks_x << "," << blocks_y << " ) threads: ( " << threads_xy << ", "<< threads_xy <<" )" << std::endl;

    dim3 threads(threads_xy, threads_xy);
    dim3 blocks(blocks_x, blocks_y);

    float *raw_point_out_re = thrust::raw_pointer_cast(&d_out_re[0]);
    float *raw_point_out_im = thrust::raw_pointer_cast(&d_out_im[0]);
    float *raw_point_in = thrust::raw_pointer_cast(&d_vec[0]);
    float *raw_point_iout_re = thrust::raw_pointer_cast(&d_iout_re[0]);
    float *raw_point_iout_im = thrust::raw_pointer_cast(&d_iout_im[0]);

    dft2d(raw_point_out_re, raw_point_out_im, raw_point_in, NUMS_COUNT_M, NUMS_COUNT_N, blocks, threads);

    idft2d(raw_point_iout_re, raw_point_iout_im, raw_point_out_re, raw_point_out_im, NUMS_COUNT_M, NUMS_COUNT_N, blocks, threads);

    thrust::copy(d_out_re.begin(), d_out_re.end(), h_out_re.begin());
    thrust::copy(d_out_im.begin(), d_out_im.end(), h_out_im.begin());
    thrust::copy(d_iout_re.begin(), d_iout_re.end(), h_iout_re.begin());
    thrust::copy(d_iout_im.begin(), d_iout_im.end(), h_iout_im.begin());

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        for (int i = 0; i < h_vec.size(); ++i)
        {
            // std::cout << "[" << i << "]: " << h_vec[i] << " ==> " << h_out_re[i] << " ==> " << h_iout_re[i] << std::endl;
            if (fabs(h_vec[i] - h_iout_re[i]) > 5e-01)
            {
                std::cout << "[" << i << "]: " << h_vec[i] << " != " << h_iout_re[i] << std::endl;
            }
        }       
    }
    else
    {
        std::cout << "cudaGetLastError: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}