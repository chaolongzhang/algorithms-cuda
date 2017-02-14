#ifndef RGR_KERNELS_CUH
#define RGR_KERNELS_CUH

#define BLOCK_DIM 16

// Transpose kernel
template<typename T>
__global__ void d_transpose(T *odata, T *idata, int width, int height)
{
    __shared__ T block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

template<typename T>
__global__ void d_forward_pass(T *out, T *in, size_t width, size_t height, float B, float b0, float b1, float b2, float b3)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= height)
    {
        return;
    }

    T x0 = 0;
    T x1 = in[0];       // in[i-1]
    T x2 = in[0];       // in[i-2]
    T x3 = in[0];       // in[i-3]
    T y = 0;
    for (int i = 0; i < width; ++i)
    {
        x0 = in[row * width + i];
        y = B * x0 + (b1 * x1 + b2 * x2 + b3 * x3) / b0;
        x3 = x2;
        x2 = x1;
        x1 = y;

        out[row * width + i] = y;
    }
}

template<typename T>
__global__ void d_backward_pass(T *out, T *in, 
    size_t width, size_t height, 
    float B, float b0, float b1, float b2, float b3)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= height)
    {
        return;
    }

    T x0 = 0;
    T x1 = in[width - 1];       // in[i+1]
    T x2 = in[width - 1];       // in[i+2]
    T x3 = in[width - 1];       // in[i+3]
    T y = 0;
    for (int i = width - 1; i >= 0; --i)
    {
        x0 = in[row * width + i];
        y = B * x0 + (b1 * x1 + b2 * x2 + b3 * x3) / b0;
        x3 = x2;
        x2 = x1;
        x1 = y;

        out[row * width + i] = y;
    }
}

#endif