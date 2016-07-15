// Two steps reduction with template-based loop expansion
template<unsigned int numThreads>
__global__ void reduction1_kernel(int *out, const int *in, size_t N)
{
    // lenght = threads (BlockDim.x)
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;
    for (size_t i = blockIdx.x * numThreads+ tid; i < N; i += numThreads * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();

    unsigned int floorPow2 = blockDim.x;
    if (floorPow2 & (floorPow2 - 1))
    {
        while(floorPow2 & (floorPow2 - 1))
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
        volatile int *wsSum = sPartials;
        if (floorPow2 >= 64) wsSum[tid] += wsSum[tid + 32];
        if (floorPow2 >= 32) wsSum[tid] += wsSum[tid + 16];
        if (floorPow2 >= 16) wsSum[tid] += wsSum[tid + 8];
        if (floorPow2 >= 8) wsSum[tid] += wsSum[tid + 4];
        if (floorPow2 >= 4) wsSum[tid] += wsSum[tid + 2];
        if (floorPow2 >= 2) wsSum[tid] += wsSum[tid + 1];

        if (tid == 0)
        {
            volatile int *wsSum = sPartials;
            out[blockIdx.x] = wsSum[0];
        }
    }
}

template<unsigned int numThreads>
void reduction1_template(int *answer, int *partial, const int *in, const size_t N, const int numBlocks)
{
    unsigned int sharedSize = numThreads * sizeof(int);

    // kernel execution
    reduction1_kernel<numThreads><<<numBlocks, numThreads, sharedSize>>>(partial, in, N);
    reduction1_kernel<numThreads><<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}

void reduction1t(int *answer, int *partial, const int *in, const size_t N, const int numBlocks, int numThreads)
{
    switch (numThreads)
    {
        case 1: reduction1_template<1>(answer, partial, in, N, numBlocks); break;
        case 2: reduction1_template<2>(answer, partial, in, N, numBlocks); break;
        case 4: reduction1_template<4>(answer, partial, in, N, numBlocks); break;
        case 8: reduction1_template<8>(answer, partial, in, N, numBlocks); break;
        case 16: reduction1_template<16>(answer, partial, in, N, numBlocks); break;
        case 32: reduction1_template<32>(answer, partial, in, N, numBlocks); break;
        case 64: reduction1_template<64>(answer, partial, in, N, numBlocks); break;
        case 128: reduction1_template<128>(answer, partial, in, N, numBlocks); break;
        case 256: reduction1_template<256>(answer, partial, in, N, numBlocks); break;
        case 512: reduction1_template<512>(answer, partial, in, N, numBlocks); break;
        case 1024: reduction1_template<1024>(answer, partial, in, N, numBlocks); break;
    }
}