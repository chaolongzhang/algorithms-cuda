
__global__ void work_inefficient_scan_kernel(float *out, const float *in, int inputSize)
{
    extern __shared__ float sXY[];

    const int tid = threadIdx.x;
    if (tid < inputSize)
    {
        sXY[tid] = in[tid];
    }

    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
    {
        __syncthreads();
        sXY[tid] += sXY[tid - stride];
    }

    out[tid] = sXY[tid];
}