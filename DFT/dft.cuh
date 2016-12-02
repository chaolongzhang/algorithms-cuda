
#define PI2 (6.283185)

__global__ void dft_kernel(float *out_re, float *out_im, const float *in, size_t N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }
    float xn = in[tid];
    for (int k = 0; k < N; ++k)
    {
        float re = xn * cosf(tid * PI2 * k / N);
        float im = -xn * sinf(tid * PI2 * k / N);
        atomicAdd(&out_re[k], re);
        atomicAdd(&out_im[k], im);
    }   
}

__global__ void dft2_kernel(float *out_re, float *out_im, const float *in, size_t N)
{
    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    float re = 0;
    float im = 0;

    for (int n = 0; n < N; ++n)
    {
        float xn = in[n];
        re += xn * cosf(n * PI2 * tid / N);
        im += -xn * sinf(n * PI2 * tid / N);
    }
  
    out_re[tid] = re;
    out_im[tid] = im;
}

__global__ void dft3_kernel(float *out_re, float *out_im, const float *in, size_t N)
{
    extern __shared__ float sIn[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
    {
        return;
    }

    sIn[tid] = in[tid];
    __syncthreads();

    float re = 0;
    float im = 0;

    for (int n = 0; n < N; ++n)
    {
        float xn = sIn[n];
        re += xn * cosf(n * PI2 * tid / N);
        im += -xn * sinf(n * PI2 * tid / N);
    }
  
    out_re[tid] = re;
    out_im[tid] = im;
}

__global__ void dft4_kernel(float *out_re, float *out_im, const float *in, size_t N)
{
    extern __shared__ float sIn[];

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int TILE_WIDTH = blockDim.x;
    if (tid >= N)
    {
        return;
    }

    float re = 0;
    float im = 0;

    for (int step = 0; step < N / TILE_WIDTH; ++step)
    {
        sIn[tx] = in[(step * TILE_WIDTH) + tx];
        __syncthreads();

        for (int ln = 0; ln < TILE_WIDTH; ++ln)
        {
            float xn = sIn[ln];
            int n = step * TILE_WIDTH + ln;
            re += xn * cosf(n * PI2 * tid / N);
            im += -xn * sinf(n * PI2 * tid / N);
        }
        __syncthreads();
    }
  
    out_re[tid] = re;
    out_im[tid] = im;
}

__global__ void idft_kernel(float *out_re, float *out_im, const float *in_re, const float *in_im, size_t N)
{
    // sIn[0:TILE_WIDTH-1] => in_re
    // sIn[N: 2TILE_WIDTH - 1] ==> in_im
    extern __shared__ float sIn[];

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int TILE_WIDTH = blockDim.x;
    if (tid >= N)
    {
        return;
    }
    int n = tid;
    float re = 0;
    float im = 0;

    for (int step = 0; step < N / TILE_WIDTH; ++step)
    {
        sIn[tx] = in_re[(step * TILE_WIDTH) + tx];
        sIn[TILE_WIDTH + tx] = in_im[(step * TILE_WIDTH) + tx];
        __syncthreads();

        for (int lk = 0; lk < TILE_WIDTH; ++lk)
        {
            int k = step * TILE_WIDTH + lk;
            re += sIn[lk] * cosf(PI2 * n * k / N) - sIn[TILE_WIDTH + lk] * sinf(PI2 * n * k / N);
            im += sIn[lk] * sinf(PI2 * n * k / N) + sIn[TILE_WIDTH + lk] * cosf(PI2 * n * k / N);
        }
        __syncthreads();
    }

    re /= N;
    im /= N;
    out_re[n] = re;
    out_im[n] = im;
}


void dft(float *out_re, float *out_im, const float *in, size_t N, const int numBlocks, const int numThreads)
{
    cudaMemset(out_re, 0, N * sizeof(float));
    cudaMemset(out_im, 0, N * sizeof(float));
    unsigned int sharedSize = numThreads * sizeof(float);
    dft2_kernel<<<numBlocks, numThreads, sharedSize>>>(out_re, out_im, in, N);
}

void idft(float *out_re, float *out_im, const float *in_re, const float *in_im, size_t N, const int numBlocks, const int numThreads)
{
    cudaMemset(out_re, 0, N * sizeof(float));
    cudaMemset(out_im, 0, N * sizeof(float));

    unsigned int sharedSize = 2 * numThreads * sizeof(float);

    idft_kernel<<<numBlocks, numThreads, sharedSize>>>(out_re, out_im, in_re, in_im, N);
}


