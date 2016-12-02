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

#include "convolvefft.cuh"

#define N_A (1024 * 1024 * 8)
#define N_B (100)
#define NUM_THREADS (256)

using namespace std;

int main(int argc, char const *argv[])
{
    size_t N = N_A + N_B - 1;
    size_t L = pow( 2, static_cast<int>(log2( static_cast<double>(N) ) + 1 ));

    thrust::host_vector<float> h_a(L, 0);
    thrust::device_vector<float> d_a(L, 0);
    thrust::host_vector<float> h_b(L, 0);
    thrust::device_vector<float> d_b(L, 0);

    thrust::host_vector<float> h_result(L);
    thrust::device_vector<float> d_result(L);

    thrust::sequence(h_a.begin(), h_a.begin() + N_A);
    thrust::sequence(h_b.begin(), h_b.begin() + N_B);
    
    // get raw pointer for kernel
    float *raw_point_a = thrust::raw_pointer_cast( &d_a[0] );
    float *raw_point_b = thrust::raw_pointer_cast( &d_b[0] );
    float *raw_point_result = thrust::raw_pointer_cast( &d_result[0] );

    int numThreads = NUM_THREADS;

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    cudaEventRecord(start);

    // copy a b to device
    thrust::copy(h_a.begin(), h_a.end(), d_a.begin());
    thrust::copy(h_b.begin(), h_b.end(), d_b.begin());

    convfft(raw_point_a, raw_point_b, raw_point_result, N, L, numThreads);

    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());

    cudaEventRecord(stop);
    checkCudaErrors( cudaThreadSynchronize() );
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cout << "run times: " << time << " ms " << endl;

    // for (thrust::host_vector<float>::iterator i = h_result.begin(); i != h_result.begin() + N; ++i)
    // {
    //     cout << *i << "\t";
    // }
    cout << endl;

    return 0;
}