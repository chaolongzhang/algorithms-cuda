// 两遍规约
__global__ void reduction1_kernel(int *out, const int *in, size_t N)
{
    // lenght = threads (BlockDim.x)
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;
    for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    unsigned int floowPow2 = blockDim.x;
    if (floowPow2 & (floowPow2 - 1))
    {
        while(floowPow2 & (floowPow2 - 1))
        {
            floowPow2 &= (floowPow2 - 1);
        }
        if (tid >= floowPow2)
        {
            sPartials[tid - floowPow2] += sPartials[tid];
        }
        __syncthreads();
    }

    for (int activeTrheads = floowPow2 / 2; activeTrheads > 32; activeTrheads /= 2)
    {
        if (tid < activeTrheads)
        {
            sPartials[tid] += sPartials[tid + activeTrheads];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int *wsSum = sPartials;
        if (floowPow2 > 32)
        {
            wsSum[tid] += wsSum[tid + 32];
        }

        if (floowPow2 > 16) wsSum[tid] += wsSum[tid + 16];
        if (floowPow2 > 8) wsSum[tid] += wsSum[tid + 8];
        if (floowPow2 > 4) wsSum[tid] += wsSum[tid + 4];
        if (floowPow2 > 2) wsSum[tid] += wsSum[tid + 2];
        if (floowPow2 > 1) wsSum[tid] += wsSum[tid + 1];

        if (tid == 0)
        {
            volatile int *wsSum = sPartials;
            out[blockIdx.x] = wsSum[0];
        }
    }
}

void reduction1(int *answer, int *partial, const int *in, const size_t N, const int numBlocks, int numThreads)
{
    unsigned int sharedSize = numThreads * sizeof(int);

    // kernel execution
    reduction1_kernel<<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
    reduction1_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}