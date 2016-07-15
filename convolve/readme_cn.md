# FFT快速卷积算法

* convolve.py Python实现的卷积算法
* convolve.cuh/convolve.cu 基于CUDA的并行卷积算法
* convolvefft.cuh/convolvefft.cu 基于cuFFT实现的CUDA并行快速FFT卷积算法

# 使用：

```shell
make
./convolve.exe
./convolvefft.exe
```

更多请参考：[CUDA并行算法系列之FFT快速卷积](http://blog.5long.me/2016/algorithms-on-cuda-fft-convolution/)