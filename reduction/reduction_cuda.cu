#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

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
    std::cout << "blocks: " << blocks << " threads: " << NUM_THREADS << std::endl;

    size_t nums_mem_size = NUM_COUNT * sizeof(int);
    int *h_nums = (int *)malloc(nums_mem_size);
    int *d_nums;
    checkCudaErrors(cudaMalloc(&d_nums, nums_mem_size));

    size_t partical_mem_size = blocks * sizeof(int);
    int *d_partial;
    checkCudaErrors(cudaMalloc(&d_partial, partical_mem_size));

    size_t answer_mem_size = 1 * sizeof(int);
    int *h_answer = (int *)malloc(answer_mem_size);
    int *d_answer;
    checkCudaErrors(cudaMalloc(&d_answer, answer_mem_size));

    for (int i = 0; i < NUM_COUNT; ++i)
    {
        h_nums[i] = 1;
    }

    int host_answer = 0;
    for (int i = 0; i < NUM_COUNT; ++i)
    {
        host_answer += h_nums[i];
    }

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

    // memoty copy
    checkCudaErrors(cudaMemcpy(d_nums, h_nums, nums_mem_size, cudaMemcpyHostToDevice));

    for (int i = 0; i < numTests; ++i)
    {
        double time = run(100, d_answer, d_partial, 
                          d_nums, NUM_COUNT, blocks, 
                          NUM_THREADS, rgTests[i].func);

        checkCudaErrors(cudaMemcpy(h_answer, d_answer, answer_mem_size, cudaMemcpyDeviceToHost));
        
        std::string equal = (host_answer == *h_answer) ? "=" : "!=";

        std::cout << rgTests[i].name <<  " time: " << time 
                  << "ms host answer (" << host_answer << ") " 
                  << equal << " device answer (" << *h_answer << ")" 
                  << std::endl;
    }

    return 0;
}