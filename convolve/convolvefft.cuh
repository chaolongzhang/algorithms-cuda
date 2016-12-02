#ifndef CONVOLVEFFT_H
#define CONVOLVEFFT_H

#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

typedef float2 Complex;

void fft(float *in, Complex *out, size_t size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void ifft(Complex *in, float *out, size_t size)
{
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, 1);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

struct complex_multiplies_functor
{
    const int N;

    complex_multiplies_functor(int _n) : N(_n) {}

    __host__ __device__ Complex operator()(const Complex &a, const Complex &b) const
    {
        Complex c;
        c.x = (a.x * b.x - a.y * b.y) / N;
        c.y = (a.x * b.y + a.y * b.x) / N;
        return c;
    }
};

void convfft( float *ina, float *inb, float *out, size_t len_out, size_t L, size_t numThreads)
{
    thrust::device_vector<Complex> d_a_fft(L);
    thrust::device_vector<Complex> d_b_fft(L);
    thrust::device_vector<Complex> d_c_fft(L);

    Complex *raw_point_a_fft = thrust::raw_pointer_cast(&d_a_fft[0]);
    Complex *raw_point_b_fft = thrust::raw_pointer_cast(&d_b_fft[0]);
    Complex *raw_point_c_fft = thrust::raw_pointer_cast(&d_c_fft[0]);

    fft(ina, raw_point_a_fft, L);
    fft(inb, raw_point_b_fft, L);

    thrust::transform(d_a_fft.begin(), d_a_fft.end(), d_b_fft.begin(), d_c_fft.begin(), complex_multiplies_functor(L));

    ifft(raw_point_c_fft, out, L);
}

#endif