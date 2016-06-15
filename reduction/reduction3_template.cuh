
template<unsigned int numThreads>
__device__ void reduction3_logStepShared(int *out, volatile int *sPartials)
{
    const int tid = threadIdx.x;
    int floorPow2 = numThreads;

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

    if (floorPow2 >= 1024)
    {
        if (tid < 512) sPartials[tid] += sPartials[tid + 512];
        __syncthreads();
    }
    if (floorPow2 >= 512)
    {
        if (tid < 256) sPartials[tid] += sPartials[tid + 256];
        __syncthreads();
    }
    if (floorPow2 >= 256)
    {
        if (tid < 128) sPartials[tid] += sPartials[tid + 128];
        __syncthreads();
    }
    if (floorPow2 >= 128)
    {
        if (tid < 64) sPartials[tid] += sPartials[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        if (floorPow2 >= 64) sPartials[tid] += sPartials[tid + 32];
        if (floorPow2 >= 32) sPartials[tid] += sPartials[tid + 16];
        if (floorPow2 >= 16) sPartials[tid] += sPartials[tid + 8];
        if (floorPow2 >= 8) sPartials[tid] += sPartials[tid + 4];
        if (floorPow2 >= 4) sPartials[tid] += sPartials[tid + 2];
        if (floorPow2 >= 2) sPartials[tid] += sPartials[tid + 1];
        if (tid == 0)
        {
            *out = sPartials[0];
        }
    }
}

__device__ unsigned int retirementCount = 0;

template<unsigned int numThreads>
__global__ void reduction3_kernel(int *out, int *partial, const int *in, size_t N)
{
    extern __shared__ int sPartials[];
    const int tid = threadIdx.x;
    int sum = 0;

    for (size_t i = blockIdx.x * numThreads + tid; i < N; i += numThreads * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    if (gridDim.x == 1)
    {
        reduction3_logStepShared<numThreads>(out, sPartials);
        return;
    }
    reduction3_logStepShared<numThreads>(&partial[blockIdx.x], sPartials);

    __shared__ bool lastBlock;
    __threadfence();

    if (tid == 0)
    {
        unsigned int ticket = atomicAdd(&retirementCount, 1);
        lastBlock = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    if (lastBlock)
    {
        sum = 0;
        for (size_t i = tid; i < gridDim.x; i += numThreads)
        {
            sum += partial[i];
        }
        sPartials[tid] = sum;
        __syncthreads();

        reduction3_logStepShared<numThreads>(out, sPartials);
        retirementCount = 0;
    }
}

template<unsigned int numThreads>
void reduction3_template(int *answer, int *partial, const int *in, const size_t N, const int numBlocks)
{
    unsigned int sharedSize = numThreads * sizeof(int);
    reduction3_kernel<numThreads><<<numBlocks, numThreads, sharedSize>>>(answer, partial, in, N);
}

void reduction3t(int *answer, int *partial, const int *in, const size_t N, const int numBlocks, int numThreads)
{
    switch (numThreads)
    {
        case 1024: reduction3_template<1024>(answer, partial, in, N, numBlocks); break;
        
        case 1: reduction3_template<1>(answer, partial, in, N, numBlocks); break;
        case 2: reduction3_template<2>(answer, partial, in, N, numBlocks); break;
        case 4: reduction3_template<4>(answer, partial, in, N, numBlocks); break;
        case 8: reduction3_template<8>(answer, partial, in, N, numBlocks); break;
        case 16: reduction3_template<16>(answer, partial, in, N, numBlocks); break;
        case 32: reduction3_template<32>(answer, partial, in, N, numBlocks); break;
        case 64: reduction3_template<64>(answer, partial, in, N, numBlocks); break;
        case 128: reduction3_template<128>(answer, partial, in, N, numBlocks); break;
        case 256: reduction3_template<256>(answer, partial, in, N, numBlocks); break;
        case 512: reduction3_template<512>(answer, partial, in, N, numBlocks); break;
        
    }
}