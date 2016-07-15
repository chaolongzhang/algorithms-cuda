
// Direct calculation convolution
__global__ void conv_kernel(const float *ina, const float *inb, float *out, size_t len_a, size_t len_b, size_t len_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len_out)
    {
        return;
    }

    float sum = 0.0f;
    for (int m = 0; m < len_b; ++m)
    {
        int k = tid - m;
        if (0 <= k && k < len_a)
        {
            sum += ina[k] * inb[m];
        }
    }
    out[tid] = sum;
}

// Optimized by shared memory
__global__ void conv2_kernel(const float *ina, const float *inb, float *out, size_t len_a, size_t len_b, size_t len_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tx = threadIdx.x;
    const size_t len_s = blockDim.x;

    extern __shared__ float sIna[];

    if (tid >= len_out)
    {
        return;
    }

    if (tid < len_a)
    {
        sIna[tx] = ina[tid];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int m = 0; m < len_b; ++m)
    {
        int k = tid - len_b + m + 1;
        int sk = tx - len_b + m + 1;
        if (0 <= sk && sk < len_s)
        {
            sum += sIna[sk] * inb[m];
        }
        else
        {
            if (0 <= k && k < len_a)
            {
                sum += ina[k] * inb[m];
            }
        }
    }
    out[tid] = sum;
}

// Optimized by shared meory and constant memory
__constant__ static float c_b[1024];
__global__ void conv3_kernel(const float *ina, float *out, size_t len_a, size_t len_b, size_t len_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tx = threadIdx.x;
    const size_t len_s = blockDim.x;

    extern __shared__ float sIna[];

    const float *inb = &c_b[0];

    if (tid >= len_out)
    {
        return;
    }

    if (tid < len_a)
    {
        sIna[tx] = ina[tid];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int m = 0; m < len_b; ++m)
    {
        int k = tid - len_b + m + 1;
        int sk = tx - len_b + m + 1;
        if (0 <= sk && sk < len_s)
        {
            sum += sIna[sk] * inb[m];
        }
        else
        {
            if (0 <= k && k < len_a)
            {
                sum += ina[k] * inb[m];
            }
        }
    }
    out[tid] = sum;
}

void conv(const float *ina, const float *inb, float *out, size_t len_a, size_t len_b, size_t len_out, size_t numBlocks, size_t numThreads)
{
    conv_kernel<<<numBlocks, numThreads>>>(ina, inb, out, len_a, len_b, len_out);
}

void conv2(const float *ina, const float *inb, float *out, size_t len_a, size_t len_b, size_t len_out, size_t numBlocks, size_t numThreads)
{   
    cudaMemcpyToSymbol(c_b, inb, len_b * sizeof(float));
    size_t sharedSize = numThreads * sizeof(float);
    conv3_kernel<<<numBlocks, numThreads, sharedSize>>>(ina, out, len_a, len_b, len_out);
}