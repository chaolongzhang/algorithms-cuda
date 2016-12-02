
#define PI2 (6.283185)


// 按列读，不能合并内存访问
__global__ void dft2d_kernel(float *out_re, float *out_im, const float *in, size_t M, size_t N)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N)
    {
        return;
    }

    float fmn = in[m * N + n];

    for ( int k = 0; k < M; ++k )
    {
        for ( int l = 0; l < N; ++l )
        {
            float temp = (float)k * m / M + (float)l * n / N;
            float re = fmn * cosf(PI2 * temp);
            float im = -fmn * sinf(PI2 * temp);
            
            atomicAdd(&out_re[k * N + l], re);
            atomicAdd(&out_im[k * N + l], im);
        }
    }
}

// 按行读，可以合并内存访问
__global__ void dft2d2_kernel(float *out_re, float *out_im, const float *in, size_t M, size_t N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N)
    {
        return;
    }

    float fmn = in[m * N + n];

    for ( int k = 0; k < M; ++k )
    {
        for ( int l = 0; l < N; ++l )
        {
            float temp = (float)k * m / M + (float)l * n / N;
            float re = fmn * cosf(PI2 * temp);
            float im = -fmn * sinf(PI2 * temp);
            
            atomicAdd(&out_re[k * N + l], re);
            atomicAdd(&out_im[k * N + l], im);
        }
    }
}

__global__ void dft2d3_kernel(float *out_re, float *out_im, const float *in, size_t M, size_t N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= M || l >= N)
    {
        return;
    }

    float rekl = 0;
    float imkl = 0;
    for ( int n = 0; n < N; ++n )
    {
        for ( int m = 0; m < M; ++m )
        {
            float fmn = in[m * N + n];
            float temp = (float)k * m / M + (float)l * n / N;
            rekl += fmn * cosf(PI2 * temp);
            imkl += -fmn * sinf(PI2 * temp);
        }
    }

    out_re[k * N + l] = rekl;
    out_im[k * N + l] = imkl;
}

// 使用共享内存进行优化
__global__ void dft2d4_kernel(float *out_re, float *out_im, const float *in, size_t M, size_t N)
{
    extern __shared__ float part[];

    int TILE_WIDTH_X = blockDim.x;
    int TILE_WIDTH_Y = blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= M || l >= N)
    {
        return;
    }

    float rekl = 0;
    float imkl = 0;

    for (int m_step = 0; m_step < M / TILE_WIDTH_Y; ++m_step)
    {
        for (int n_step = 0; n_step < N / TILE_WIDTH_X; ++n_step)
        {
            int row = m_step * TILE_WIDTH_Y + ty;
            int col = n_step * TILE_WIDTH_X + tx;
            part[ty * TILE_WIDTH_X + tx] = in[row * N + col];
            __syncthreads();

            for (int lm = 0; lm < TILE_WIDTH_Y; ++lm)
            {
                for (int ln = 0; ln < TILE_WIDTH_X; ++ln)
                {
                    int m = m_step * TILE_WIDTH_Y + lm;
                    int n = n_step * TILE_WIDTH_X + ln;
                    float fmn = part[lm * TILE_WIDTH_X + ln];
                    float temp = (float)k * m / M + (float)l * n / N;
                    rekl += fmn * cosf(PI2 * temp);
                    imkl += -fmn * sinf(PI2 * temp);
                }
            }

            __syncthreads();
        }
    }

    out_re[k * N + l] = rekl;
    out_im[k * N + l] = imkl;
}

__global__ void idft2d_kernel(float *out_re, float *out_im, const float *in_re, const float *in_im, size_t M, size_t N)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N)
    {
        return;
    }

    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k < M; ++k)
    {
        for (int l = 0; l < N; ++l)
        {
            float temp = (float)k * m / M + (float)l * n / N;
            float temp_cos = cosf(PI2 * temp);
            float temp_sin = sinf(PI2 * temp);
            re += in_re[k * N + l] * temp_cos - in_im[k * N + l] * temp_sin;
            im += in_re[k * N + l] * temp_sin + in_im[k * N + l] * temp_cos;
        }
    }
    re /= (M * N);
    im /= (M * N);
    out_re[m * N + n] = re;
    out_im[m * N + n] = im;
}

__global__ void idft2d2_kernel(float *out_re, float *out_im, const float *in_re, const float *in_im, size_t M, size_t N)
{
    extern __shared__ float part[];

    int TILE_WIDTH_X = blockDim.x;
    int TILE_WIDTH_Y = blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N)
    {
        return;
    }

    float re = 0.0f;
    float im = 0.0f;

    for (int k_step = 0; k_step < M / TILE_WIDTH_Y; ++k_step)
    {
        for (int l_step = 0; l_step < N / TILE_WIDTH_X; ++l_step)
        {
            int row = k_step * TILE_WIDTH_Y + ty;
            int col = l_step * TILE_WIDTH_X + tx;
            part[ty * 2 * TILE_WIDTH_X + tx] = in_re[row * N + col];
            part[ty * 2 * TILE_WIDTH_X + TILE_WIDTH_X + tx] = in_im[row * N + col];
            __syncthreads();

            // re = part[ty * 2 * TILE_WIDTH_X + tx];
            // im = part[ty * 2 * TILE_WIDTH_X + TILE_WIDTH_X + tx];

            for (int lk = 0; lk < TILE_WIDTH_Y; ++lk)
            {
                for (int ll = 0; ll < TILE_WIDTH_X; ++ll)
                {
                    int k = k_step * TILE_WIDTH_Y + lk;
                    int l = l_step * TILE_WIDTH_X + ll;
                    float temp = (float)k * m / M + (float)l * n / N;
                    float temp_cos = cosf(PI2 * temp);
                    float temp_sin = sinf(PI2 * temp);
                    re += part[lk * 2 * TILE_WIDTH_X + ll] * temp_cos - part[lk * 2 * TILE_WIDTH_X + TILE_WIDTH_X + ll] * temp_sin;
                    im += part[lk * 2 * TILE_WIDTH_X + ll] * temp_sin + part[lk * 2 * TILE_WIDTH_X + TILE_WIDTH_X + ll] * temp_cos;
                }
            }
            __syncthreads();
        }
    }
    re /= (M * N);
    im /= (M * N);
    out_re[m * N + n] = re;
    out_im[m * N + n] = im;
}

void dft2d(float *out_re, float *out_im, const float *in, size_t M, size_t N, const dim3 numBlocks, const dim3 numThreads)
{
    cudaMemset(out_re, 0, M * N * sizeof(float));
    cudaMemset(out_im, 0, M * N * sizeof(float));

    size_t sharedSize = numThreads.x * numThreads.y * sizeof(float);

    // dft2d3_kernel<<<numBlocks, numThreads>>>(out_re, out_im, in, M, N);

    dft2d4_kernel<<<numBlocks, numThreads, sharedSize>>>(out_re, out_im, in, M, N);
}


void idft2d(float *out_re, float *out_im, const float *in_re, const float *in_im, size_t M, size_t N, const dim3 numBlocks, const dim3 numThreads)
{
    size_t sharedSize = 2 * numThreads.x * numThreads.y * sizeof(float);
    idft2d2_kernel<<<numBlocks, numThreads, sharedSize>>>(out_re, out_im, in_re, in_im, M, N);
}



