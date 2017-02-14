#include <windows.h>
#include <cuda_runtime.h>

#include "timeclock.h"

void CPU_Timeclock::start()
{
    start_t = GetTickCount();
}

void CPU_Timeclock::record()
{
    stop_t = GetTickCount();
}

float CPU_Timeclock::elapsed_time()
{
    return (float)(stop_t - start_t);
}

#ifdef GPU

void GPU_Timeclock::start()
{
    cudaEventRecord(this->start_t, 0);
}

void GPU_Timeclock::record()
{
    cudaEventRecord(this->stop_t, 0);
    cudaDeviceSynchronize();
}

float GPU_Timeclock::elapsed_time()
{
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, this->start_t, this->stop_t);
    return elapsedTime;
}

#endif