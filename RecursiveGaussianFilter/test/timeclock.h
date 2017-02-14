#ifndef TIME_CLOCK_H
#define TIME_CLOCK_H

#include <windows.h>

class CPU_Timeclock
{
public:
    CPU_Timeclock() {}
    ~CPU_Timeclock() {}
    
    void start();
    void record();
    float elapsed_time();

private:
    DWORD start_t, stop_t;
};

#ifdef GPU

class GPU_Timeclock
{
public:
    GPU_Timeclock()
    {
        cudaEventCreate(&start_t);
        cudaEventCreate(&stop_t);
    }
    ~GPU_Timeclock()
    {
        cudaEventDestroy(start_t);
        cudaEventDestroy(stop_t);
    }

    void start();
    void record();
    float elapsed_time();

private:
    cudaEvent_t start_t, stop_t;    
};

#endif

#endif