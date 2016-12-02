#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#include <iostream>

#include "scan1.cuh"

#define N (32)
#define NUM_THREADS (N)

int main(int argc, char const *argv[])
{
    
    thrust::host_vector<int> h_vec(N);
    thrust::device_vector<int> d_vec(N);
    thrust::device_vector<int> h_answer(N);
    thrust::device_vector<int> d_answer(N);

    thrust::fill(h_vec.begin(), h_vec.end(), 1);
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_answer.begin());

    unsigned int numThreads = NUM_THREADS;
    unsigned int numBlocks = (N + numThreads - 1) / numThreads;

    int *raw_point_nums = thrust::raw_pointer_cast(&d_vec[0]);

    scan1(raw_point_nums, N, numBlocks, numThreads);
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    thrust::copy(d_answer.begin(), d_answer.end(), h_answer.begin());

    bool success = true;
    for (int i = 0; i < h_vec.size(); ++i)
    {
        if (h_vec[i] - d_answer[i] != 0)
        {
            std::cout << "i = " << i << " " <<  h_vec[i] << "!=" << d_answer[i] << "\t";
            success = false;
        }
        
    }
    std::cout << std::endl;

    if (success)
    {
        std::cout << "done" << std::endl;
    }

    return 0;
}