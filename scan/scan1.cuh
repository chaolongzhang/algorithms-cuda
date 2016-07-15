
__global__ void scan1_kernel(int *nums, size_t N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) { return; }

    __shared__ int lgN;

    if (tid == 0)
    {
        lgN = log2f(N);
    }

    // Bottom-up
    for (int d = 0; d < lgN; ++d)
    {
        int d21 = powf(2, d + 1);
        int d2 = d21 / 2;
        if (tid % d21 == d21 - 1)
        {
            nums[tid] += nums[tid - d2];
        }
        __syncthreads();
    }

    // top down
    if (tid == 0)
    {
        nums[N - 1] = 0;
    }
    __syncthreads();

    for (int d = lgN - 1; d >= 0; --d)
    {
        int d21 = powf(2, d + 1);
        int d2 = d21 / 2;
        if (tid % d21 == d21 - 1)
        {
            int temp = nums[tid - d2];
            nums[tid - d2] = nums[tid];
            nums[tid] += temp;
        }
        __syncthreads();
    }
}

void scan1(int *nums, size_t N, unsigned int numBlocks, unsigned int numThreads)
{
    scan1_kernel<<<numBlocks, numThreads>>>(nums, N);
}