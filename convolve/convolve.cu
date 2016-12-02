#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#include <iostream>
#include <cmath>

#include "convolve.cuh"

#define N_A (1024 * 1024 * 8)
#define N_B (100)
#define NUM_THREADS (256)

using namespace std;

int main(int argc, char const *argv[])
{
    thrust::host_vector<float> h_a(N_A);
    thrust::device_vector<float> d_a(N_A);
    thrust::host_vector<float> h_b(N_B);
    thrust::device_vector<float> d_b(N_B);

    size_t N = h_a.size() + h_b.size() - 1;
    size_t L = pow( 2, static_cast<int>(log2(N - 1)) + 1 );

    thrust::host_vector<float> h_result(N);
    thrust::device_vector<float> d_result(N);

    thrust::sequence(h_a.begin(), h_a.end());
    // thrust::sequence(h_b.begin(), h_b.end());
    thrust::sequence(h_b.rbegin(), h_b.rend());
    
    // get raw pointer for kernel
    float *raw_point_a = thrust::raw_pointer_cast( &d_a[0] );
    float *raw_point_b = thrust::raw_pointer_cast( &d_b[0] );
    float *raw_point_result = thrust::raw_pointer_cast( &d_result[0] );

    int numThreads = NUM_THREADS;
    int numBlocks = (L + numThreads - 1) / numThreads;

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    cudaEventRecord(start);

    // copy a b to device
    thrust::copy(h_a.begin(), h_a.end(), d_a.begin());
    thrust::copy(h_b.begin(), h_b.end(), d_b.begin());

    // conv(raw_point_a, raw_point_b, raw_point_result, N_A, N_B, L, numBlocks, numThreads);
    conv2(raw_point_a, raw_point_b, raw_point_result, N_A, N_B, N, numBlocks, numThreads);

    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());

    cudaEventRecord(stop);
    checkCudaErrors( cudaThreadSynchronize() );
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cout << "run times: " << time << " ms " << endl;

    // for (thrust::host_vector<float>::iterator i = h_result.begin(); i != h_result.end(); ++i)
    // {
    //     cout << *i << "\t";
    // }
    cout << endl;

    return 0;
}