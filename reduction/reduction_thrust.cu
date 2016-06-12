#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <iostream>

#include "reduction1.h"
#include "reduction1_template.h"
#include "reduction2.h"
#include "reduction2_template.h"
#include "reduction3.h"
#include "reduction3_template.h"

typedef void (*pfnReduction)(int*, int*, const int*, size_t, int, int);

#define NUM_COUNT (1024 * 1024 * 1)
#define NUM_THREADS (1024)
#define MAX_BLOCKS (1024)

double run(int cIterations, int *answer, int *partial, const int *in, const size_t N, 
    const int numBlocks, int numThreads, pfnReduction func)
{
    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&stop) );
    cudaEventRecord(start);

    for (int i = 0; i < cIterations; ++i)
    {
        func(answer, partial, in, N, numBlocks, numThreads);
    }

    cudaEventRecord(stop);
    checkCudaErrors( cudaThreadSynchronize() );
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    return time;
}

int main(int argc, char const *argv[])
{
    int blocks = (NUM_COUNT + NUM_THREADS - 1) / NUM_THREADS;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    std::cout << "blocks: " << blocks << " threads: " << NUM_THREADS << std::endl;

    thrust::host_vector<int> h_vec(NUM_COUNT);
    thrust::fill(h_vec.begin(), h_vec.end(), 1);

    thrust::device_vector<int> d_vec(NUM_COUNT);
    thrust::device_vector<int> d_answer(1);
    thrust::device_vector<int> d_partial(blocks);

    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    int *raw_point_nums = thrust::raw_pointer_cast(&d_vec[0]);
    int *raw_point_partial = thrust::raw_pointer_cast(&d_partial[0]);
    int *raw_point_answer = thrust::raw_pointer_cast(&d_answer[0]);

    struct
    {
        std::string name;
        pfnReduction func;
    } rgTests[] = {
        { "simple loop", reduction1 },
        { "simple loop template", reduction1t },
        { "atomicAdd", reduction2 },
        { "atomicAdd template", reduction2t },
        { "single pass", reduction3 },
        { "single pass template", reduction3t },        
    };

    int numTests = sizeof(rgTests) / sizeof(rgTests[0]);
    int host_answer = thrust::reduce(h_vec.begin(), h_vec.end());
    for (int i = 0; i < numTests; ++i)
    {
        double time = run(100, raw_point_answer, raw_point_partial, 
                          raw_point_nums, NUM_COUNT, blocks, 
                          NUM_THREADS, rgTests[i].func);
        int h_answer = d_answer[0];
        
        std::string equal = (host_answer == h_answer) ? "=" : "!=";

        std::cout << rgTests[i].name <<  " time: " << time 
                  << "ms host answer (" << host_answer << ") " 
                  << equal << " device answer (" << h_answer << ")" 
                  << std::endl;
    }

    return 0;
}