# -*- coding: utf-8 -*-

import math
import numpy

PI2 = math.pi * 2


def dft2d(nums):
    """ 2D DFT
    
    args:
        nums = [ [ 1, 2, 3 ], [ ... ], ... ]
    """
    M = len(nums)
    N = len(nums[0])
    x = [ [ 0.0 + 0.0j for y in range(N) ] for x in range(M) ]

    for m in range(M):
        for n in range(N):
            for k in range(M):
                for l in range(N):
                    re = nums[m][n] * math.cos(PI2 * (k * m / M + l * n / N))
                    im = -nums[m][n] * math.sin(PI2 * (k * m / M + l * n / N))
                    x[k][l] += complex(re, im)
    return x




def idft2d(F):
    """ 2D IDFT

    args:
        F: 复数，2D傅里叶变换的结果

    args:
        二维复数
    """
    M = len(F)
    N = len(F[0])
    f = [ [ 0.0 + 0.0j for y in range(N) ] for x in range(M) ]
    MN1 = 1 / (M * N)
    for m in range(M):
        for n in range(N):
            for k in range(M):
                for l in range(N):
                    re = F[k][l].real * math.cos(PI2 * (k * m / M + l * n / N)) - F[k][l].imag * math.sin(PI2 * (k * m / M + l * n / N))
                    im = F[k][l].real * math.sin(PI2 * (k * m / M + l * n / N)) + F[k][l].imag * math.cos(PI2 * (k * m / M + l * n / N))
                    re *= MN1
                    im *= MN1
                    f[m][n] += complex(re, im)
    return f


if __name__ == '__main__':
    M = 2
    N = 2
    nums = [ [ x + 1 for y in range(N) ] for x in range(M) ]

    dft2d_result = dft2d(nums)

    # numpy_result = numpy.fft.fft2(nums)

    idft2d_result = idft2d(dft2d_result)

    # numpy_ifft = numpy.fft.ifft2(dft2d_result)

    # print("------nums--------")
    # print(nums)

    print("------dft2d--------")
    print(dft2d_result)

    # print("------numpy fft2--------")
    # print(numpy_result)

    print("------idft2d--------")
    print(idft2d_result)

    # print("------numpy ifft2--------")
    # print(numpy_ifft)

    success = True
    for m in range(M):
        for n in range(N):
            if math.fabs(nums[m][n].real - idft2d_result[m][n].real) > 1e-05:
                print("[%d,%d]: %f != %f" % (m, n, nums[m][n], idft2d_result[m][n]))
                success = False
    print("success: %s\n" % str(success))



