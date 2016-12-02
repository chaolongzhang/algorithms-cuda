__device__ void reduction3_logStepShared(int *out, volatile int *sPartials)
{
    const int tid = threadIdx.x;
    int floorPow2 = blockDim.x;

    if (floorPow2 & (floorPow2 - 1))
    {
        while (floorPow2 & (floorPow2 - 1))
        {
            floorPow2 &= (floorPow2 - 1);
        }
        if (tid >= floorPow2)
        {
            sPartials[tid - floorPow2] += sPartials[tid];
        }
        __syncthreads();
    }

    for (int activeTrheads = floorPow2 / 2; activeTrheads > 32; activeTrheads /= 2)
    {
        if (tid < activeTrheads)
        {
            sPartials[tid] += sPartials[tid + activeTrheads];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        if (floorPow2 > 32) sPartials[tid] += sPartials[tid + 32];
        if (floorPow2 > 16) sPartials[tid] += sPartials[tid + 16];
        if (floorPow2 > 8) sPartials[tid] += sPartials[8];
        if (floorPow2 > 4) sPartials[tid] += sPartials[4];
        if (floorPow2 > 2) sPartials[tid] += sPartials[2];
        if (floorPow2 > 1) sPartials[tid] += sPartials[1];

        if (tid == 0)
        {
            *out = sPartials[0];
        }
    }
}

__device__ unsigned int retirementCount_1 = 0;
__global__ void reduction3_kernel(int *out, int *partial, const int *in, size_t N)
{
    extern __shared__ int sPartials[];
    const int tid = threadIdx.x;
    int sum = 0;

    for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1)
    {
        reduction3_logStepShared(out, sPartials);
        return;
    }
    reduction3_logStepShared(&partial[blockIdx.x], sPartials);

    __shared__ bool lastBlock;
    __threadfence();

    if (tid == 0)
    {
        unsigned int ticket = atomicAdd(&retirementCount_1, 1);
        lastBlock = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    if (lastBlock)
    {
        sum = 0;
        for (size_t i = tid; i < gridDim.x; i += blockDim.x)
        {
            sum += partial[i];
        }
        sPartials[tid] = sum;
        __syncthreads();

        reduction3_logStepShared(out, sPartials);
        retirementCount_1 = 0;
    }
}

void reduction3(int *answer, int *partial, const int *in, const size_t N, const int numBlocks, int numThreads)
{
    unsigned int sharedSize = numThreads * sizeof(int);
    reduction3_kernel<<<numBlocks, numThreads, sharedSize>>>(answer, partial, in, N);
}