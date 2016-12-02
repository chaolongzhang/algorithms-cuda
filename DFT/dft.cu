#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <iostream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include "dft.cuh"

#define NUMS_COUNT (1024 * 2)
#define NUM_THREADS (256)

int main(int argc, char const *argv[])
{
    thrust::host_vector<float> h_vec(NUMS_COUNT);
    thrust::host_vector<float> h_out_re(NUMS_COUNT);
    thrust::host_vector<float> h_out_im(NUMS_COUNT);
    thrust::host_vector<float> h_iout_re(NUMS_COUNT);
    thrust::host_vector<float> h_iout_im(NUMS_COUNT);

    thrust::device_vector<float> d_vec(NUMS_COUNT);
    thrust::device_vector<float> d_out_re(NUMS_COUNT);
    thrust::device_vector<float> d_out_im(NUMS_COUNT);
    thrust::device_vector<float> d_iout_re(NUMS_COUNT);
    thrust::device_vector<float> d_iout_im(NUMS_COUNT);

    thrust::sequence(h_vec.begin(), h_vec.end());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
    thrust::fill(d_out_re.begin(), d_out_re.end(), 0);
    thrust::fill(d_out_im.begin(), d_out_im.end(), 0);

    int threads = NUM_THREADS;
    int blocks = (NUMS_COUNT + NUM_THREADS - 1) / NUM_THREADS;

    float *raw_point_out_re = thrust::raw_pointer_cast(&d_out_re[0]);
    float *raw_point_out_im = thrust::raw_pointer_cast(&d_out_im[0]);
    float *raw_point_in = thrust::raw_pointer_cast(&d_vec[0]);
    float *raw_point_iout_re = thrust::raw_pointer_cast(&d_iout_re[0]);
    float *raw_point_iout_im = thrust::raw_pointer_cast(&d_iout_im[0]);

    dft(raw_point_out_re, raw_point_out_im, raw_point_in, NUMS_COUNT, blocks, threads);

    idft(raw_point_iout_re, raw_point_iout_im, raw_point_out_re, raw_point_out_im, NUMS_COUNT, blocks, threads);

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