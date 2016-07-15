# Fast convolution algorithm based on FFT

* convolve.py Convolution algorithm implemented in Python.
* convolve.cuh/convolve.cu CUDA-based parallel convolution algorithm.
* convolvefft.cuh/convolvefft.cu Fast convolution algorithm implemented CUDA cuFFT.

# Usage

```shell
make
./convolve.exe
./convolvefft.exe
```

See [Makefile](Makefile) for detail.

# More

For more information, read my Blog [CUDA并行算法系列之FFT快速卷积](http://blog.5long.me/2016/algorithms-on-cuda-fft-convolution/).

如果你会中文，请查看[中文版](readme_cn.md).