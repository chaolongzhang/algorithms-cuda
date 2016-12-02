
template<class T>
inline __device__ T 
scanWarp( volatile T *sPartials )
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if ( lane >=  1 ) sPartials[0] += sPartials[- 1];
    if ( lane >=  2 ) sPartials[0] += sPartials[- 2];
    if ( lane >=  4 ) sPartials[0] += sPartials[- 4];
    if ( lane >=  8 ) sPartials[0] += sPartials[- 8];
    if ( lane >= 16 ) sPartials[0] += sPartials[-16];
    return sPartials[0];
}

template<class T>
inline __device__ T scanBlock(volatile T *sPartials)
{
    extern __shared__ T warpPartials[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    T sum = scanWarp<T>(sPartials);
    __syncthreads();

    if (lane == 31)
    {
        warpPartials[16 + warpid] = sum;
    }
    __syncthreads();

    if (warpid == 0)
    {
        scanWarp<T>(16 + warpPartials + tid);
    }
    __syncthreads();

    if (warpid > 0）
    {
        sum += warpPartials[16 + warpid - 1];
    }
    __syncthreads();

    *sPartials = sum;
    __syncthreads();

    return sum;
}

template<class T, bool bWriteSpine>
__global__ void scanAndWritePartials(T *out, T *gPartials, const T *in, size_t N, size_t numBlocks)
{
    extern volatile __shared__ T sPartials[];
    const int tid = threadIdx.x;
    volatile T *myShared = sPartials + tid;

    for (size_t iBlock = blockIdx.x; iBlock < numBlocks; iBlock += gridDim.x)
    {
        size_t index = iBlock * blockDim.x + tid;
        *myShared = (index < N) ? in[index] : 0;
        __syncthreads();

        T sum = scanBlock(myShared);
        __syncthreads();

        if (index < N)
        {
            out[index] = *myShared;
        }
        if (bWriteSpine && (threadIdx.x == (blockDim.x - 1)))
        {
            gPartials[iBlock] = sum;
        }
    }
}

template<class T>
__global__ void scanAddBaseSums(T *out, T *gBaseSums, size_t N, size_t numBlocks)
{
    const int tid = threadIdx.x;

    T fan_value = 0;
    for (size_t iBlock = blockIdx.x; iBlock < numBlocks; iBlock += gridDim.x)
    {
        size_t index = iBlock * blockDim.x + tid;
        if (iBlock > 0)
        {
            fan_value = gBaseSums[iBlock - 1];
        }
        out[index] += fan_value;
    }
}

template<class T>
void sacnFan(T *out, const T *in, size_t N, int b)
{
    cudaError_t status;
    if (N <= b)
    {
        scanAndWritePartials<T, false><<<1, b, b * sizeof(T)>>>(out, 0, in, N, 1);
        return;
    }

    T *gPartials = 0;

    size_t numPartials = (N) / b;

    const unsigned int maxBlocks = 150;
    unsigned int numBlocks = min( numPartials, maxBlocks );
    checkCudaErrors( cudaMalloc( &gPartials, numPartials * sizeof(T) ) );
    scanAndWritePartials<T, true><<<numBlocks, b, b * sizeof(T)>>>(out, gPartials, in, N, numPartials);
    sacnFan<T>( gPartials, gPartials, numPartials, b );
    scanAddBaseSums<T><<<numBlocks, b>>>( out, gPartials, N, numPartials );
    cudaFree( gPartials );
}

