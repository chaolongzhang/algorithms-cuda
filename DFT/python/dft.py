# -*- coding: utf-8 -*-

import math
import numpy

PI2 = math.pi * 2


def dft(nums):
    """
    离散傅里叶变换
    """
    N = len(nums)
    x = [0 + 0j for i in range(N)]
    for n in range(N):
        xn = nums[n]
        for k in range(N):
            re = xn * math.cos(n * PI2 * k / N)   # 实部
            im = -xn * math.sin(n * PI2 * k / N)  # 虚部
            x[k] += complex(re, im)
    return x

def dft2(nums):
    N = len(nums)
    x = [0 + 0j for i in range(N)]
    for k in range(N):
        for n in range(N):
            xn = nums[n]
            re = xn * math.cos(n * PI2 * k / N)
            im = -xn * math.sin(n * PI2 * k / N)
            x[k] += complex(re, im)
    return x


def idft(nums):
    """ 逆傅里叶变换
    args:
        nums: 复数，傅里叶变换的结果
    """
    N = len(nums)
    x = [0 + 0j for i in range(N)]

    for n in range(N):
        for k in range(N):
            re = nums[k].real * math.cos(PI2 * n * k / N) - nums[k].imag * math.sin(PI2 * n * k / N)
            im = nums[k].real * math.sin(PI2 * n * k / N) + nums[k].imag * math.cos(PI2 * n * k / N)
            re /= N
            im /= N
            x[n] += complex(re, im)

    return x

def idft2(nums):
    N = len(nums)
    x = [0 + 0j for i in range(N)]

    for k in range(N):
        for n in range(N):
            re = nums[k].real * math.cos(PI2 * n * k / N) - nums[k].imag * math.sin(PI2 * n * k / N)
            im = nums[k].real * math.sin(PI2 * n * k / N) + nums[k].imag * math.cos(PI2 * n * k / N)
            re /= N
            im /= N
            x[n] += complex(re, im)

    return x


if __name__ == '__main__':
    N = 1024 * 8
    nums = [i for i in range(N)]
    np_result = numpy.fft.fft(nums)
    dft_result = dft2(nums)

    inp_result = numpy.fft.ifft(dft_result)
    ix = idft2(dft_result)

    # print("------nums--------")
    # print(nums)

    # print("------numpy fft--------")
    # print(np_result)
    # print("------dft--------")
    # print(dft_result)

    # print("------numpy ifft--------")
    # print(inp_result)

    # print("------idft--------")
    # print(ix)

    success = True
    for i in range(N):
        if math.fabs(nums[i].real - ix[i].real) > 1e-05:
            print("[%d]: %f != %f" % (i, nums[i], ix[i]))
            success = False
    print("success: %s\n" % str(success))
